"""
彩票图片 OCR 识别模块 v3
功能：上传彩票图片，自动识别彩票类型、期号、购买号码（支持多注）
依赖：rapidocr_onnxruntime, Pillow
说明：v3 改用 rapidocr（纯 onnxruntime）替代 cnocr，彻底摆脱 torch 依赖
"""
# ── 最优先：修复 exe 环境下可能缺失的环境变量 ──────────────────────────────
import os as _os
if not _os.environ.get('USERNAME') and not _os.environ.get('USER'):
    _up = _os.environ.get('USERPROFILE', '')
    _uname = _os.path.basename(_up) if _up else 'user'
    _os.environ['USERNAME'] = _uname or 'user'
    _os.environ['USER'] = _os.environ['USERNAME']
_os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', _os.path.join(
    _os.environ.get('TEMP') or _os.environ.get('TMP') or _os.path.expanduser('~'),
    'torch_inductor_cache'))
_os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
del _os
# ────────────────────────────────────────────────────────────────────────────

import os
import re
import sys
import threading
import sqlite3
import tempfile
from typing import Optional, Tuple, List, Dict, Any

# 全局 OCR 引擎（延迟初始化）
_ocr_engine = None
_ocr_init_error = None


def get_ocr_engine():
    """获取或初始化 OCR 引擎（延迟加载 rapidocr）"""
    global _ocr_engine, _ocr_init_error

    if _ocr_engine is None and _ocr_init_error is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            print("[OCR] RapidOCR 初始化成功")
        except ImportError as e:
            _ocr_init_error = f"缺少 RapidOCR: {str(e)}"
            print(f"[OCR] {_ocr_init_error}")
        except Exception as e:
            _ocr_init_error = f"RapidOCR 初始化失败: {str(e)}"
            print(f"[OCR] {_ocr_init_error}")
            import traceback
            traceback.print_exc()

    return _ocr_engine


# ─────────────────────────────────────────────
# 图片预处理
# ─────────────────────────────────────────────

def _preprocess_image(img):
    """
    对彩票图片做预处理：
      1. 缩放至合适宽度（过小的图放大，过大的等比缩小）
      2. 转为 RGB
      3. 增强对比度 & 锐化
    返回处理后的 PIL.Image
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        return img

    if img.mode != 'RGB':
        img = img.convert('RGB')

    target_w = 1280
    w, h = img.size
    if w < 600:
        scale = target_w / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif w > 2000:
        scale = target_w / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


# ─────────────────────────────────────────────
# 文本解析工具
# ─────────────────────────────────────────────

def _extract_issue_from_text(text: str) -> Optional[str]:
    """从识别文本中提取期号（支持多种格式）"""
    patterns = [
        r'第\s*(\d{5,7})\s*期',
        r'期\s*号[：:]\s*(\d{5,7})',
        r'(\d{7})\s*期',
        r'(\d{6})\s*期',
        r'(\d{5})\s*期',
        r'第\s*(\d{4})\s*期',
        r'\b(2026\d{3})\b',
        r'\b(2025\d{3})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            issue = match.group(1)
            if len(issue) == 7:
                issue = issue[2:]
            elif len(issue) == 6:
                issue = issue[1:]
            elif len(issue) == 4:
                issue = '0' + issue
            return issue.zfill(5)
    return None


def _classify_lottery_type(text: str) -> Optional[str]:
    """根据识别文本判断彩票类型"""
    text_lower = text.replace(' ', '')

    if any(kw in text_lower for kw in ['双色球', '双色', 'SSQ', 'ssq', '福彩3d', '中国福利彩票双']):
        return 'ssq'
    if any(kw in text_lower for kw in ['大乐透', '超级大乐透', '大乐', 'DLT', 'dlt', '体彩大']):
        return 'dlt'
    if '体育彩票' in text_lower or '体彩' in text_lower:
        return 'dlt'
    if '福利彩票' in text_lower or '福彩' in text_lower:
        return 'ssq'
    return None


def _extract_numbers_from_line(line: str) -> List[int]:
    """从单行文本提取所有 1-99 的数字（保留原始顺序）"""
    line = re.sub(r'[,，、/\\|+＋]', ' ', line)
    nums = []
    for m in re.finditer(r'\b(\d{1,2})\b', line):
        n = int(m.group(1))
        if 1 <= n <= 99:
            nums.append(n)
    return nums


def _try_parse_ssq_line(nums: List[int]) -> Optional[Tuple[List[int], int]]:
    """尝试从数字列表解析双色球一注：红球 6 个(1-33) + 蓝球 1 个(1-16)"""
    if len(nums) < 7:
        return None

    for split in range(6, len(nums)):
        reds = nums[:split]
        blues_cand = nums[split:]
        unique_reds = sorted(set(r for r in reds if 1 <= r <= 33))
        if len(unique_reds) < 6:
            continue
        valid_blues = [b for b in blues_cand if 1 <= b <= 16]
        if valid_blues:
            return sorted(unique_reds[:6]), valid_blues[0]

    reds_cand = [n for n in nums if 1 <= n <= 33]
    blues_cand = [n for n in nums if 1 <= n <= 16]
    if len(reds_cand) >= 6 and len(blues_cand) >= 1:
        unique_reds = sorted(set(reds_cand))
        if len(unique_reds) >= 6:
            return sorted(unique_reds[:6]), blues_cand[-1]

    return None


def _try_parse_dlt_line(nums: List[int]) -> Optional[Tuple[List[int], List[int]]]:
    """尝试从数字列表解析大乐透一注：前区 5 个(1-35) + 后区 2 个(1-12)"""
    if len(nums) < 7:
        return None

    for split in range(5, len(nums) - 1):
        fronts = nums[:split]
        backs = nums[split:]
        unique_front = sorted(set(f for f in fronts if 1 <= f <= 35))
        unique_back = sorted(set(b for b in backs if 1 <= b <= 12))
        if len(unique_front) >= 5 and len(unique_back) >= 2:
            return sorted(unique_front[:5]), sorted(unique_back[:2])

    front_cand = [n for n in nums if 1 <= n <= 35]
    back_cand = [n for n in nums if 1 <= n <= 12]
    unique_front = sorted(set(front_cand))
    unique_back = sorted(set(back_cand))
    if len(unique_front) >= 5 and len(unique_back) >= 2:
        return sorted(unique_front[:5]), sorted(unique_back[:2])

    return None


def _parse_line_with_separator(line: str) -> Optional[Dict]:
    """尝试用分隔符（+ | 红/前 后 -）解析一行号码"""
    line = line.replace('＋', '+').replace('｜', '|')

    for sep in ['+', '|', '红球', '蓝球', '前区', '后区']:
        if sep in line:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                left_nums = _extract_numbers_from_line(parts[0])
                right_nums = _extract_numbers_from_line(parts[1])

                if len(left_nums) >= 6 and len(right_nums) >= 1:
                    reds = sorted(set(n for n in left_nums if 1 <= n <= 33))
                    blues = [n for n in right_nums if 1 <= n <= 16]
                    if len(reds) >= 6 and blues:
                        return {'type': 'ssq', 'reds': sorted(reds[:6]), 'blues': [blues[0]]}

                if len(left_nums) >= 5 and len(right_nums) >= 2:
                    front = sorted(set(n for n in left_nums if 1 <= n <= 35))
                    back = sorted(set(n for n in right_nums if 1 <= n <= 12))
                    if len(front) >= 5 and len(back) >= 2:
                        return {'type': 'dlt', 'reds': sorted(front[:5]), 'blues': sorted(back[:2])}

    return None


def _parse_ssq_dash_format(line: str) -> Optional[Dict]:
    """
    专门解析双色球 "红球-蓝球" 格式，例如：
      "02 06 10 16 27 28-06"
      "② 02 06 10 16 27 28-06"
    识别模式：末尾数字（1-16）是蓝球，前面 6 个数字（1-33）是红球
    """
    # 移除序号：①②③④⑤⑥⑦⑧⑨⑩（不匹配普通数字如 02）
    line = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*[.、]?\s*', '', line).strip()
    if not line:
        return None

    # 找最后一个数字（一定是蓝球），忽略数字间的 -
    # 把所有非数字字符替换为空格，再提取数字
    nums_raw = re.sub(r'[^\d\s]', ' ', line)
    nums_all = nums_raw.split()
    if len(nums_all) < 7:
        return None

    blue = int(nums_all[-1])
    if blue > 16:  # 蓝球必须是 1-16
        return None

    # 前面至少6个数字作为红球
    reds = [int(n) for n in nums_all[:-1]]
    valid_reds = sorted(set(n for n in reds if 1 <= n <= 33))
    if len(valid_reds) < 6:
        return None

    return {'type': 'ssq', 'reds': valid_reds[:6], 'blues': [blue]}


def _parse_ssq_dash_format_multi(line: str) -> List[Dict]:
    """
    解析一行中的多个 "红球-蓝球" 组合
    彩票每注格式如："06 10 16 27 28-06"，每注之间用空格或换行分开
    """
    tickets = []

    # 移除序号
    clean_line = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*[.、]?\s*', '', line).strip()
    if not clean_line:
        return []

    # 先用空格分割各部分
    parts = clean_line.split()
    if not parts:
        return []

    # 尝试：把 "-" 替换为空格，然后每7个连续数字为一注
    # 处理形如 "02 06 10 16 27 28-06" -> "02 06 10 16 27 28 06"
    normalized = re.sub(r'[\-－]', ' ', clean_line)
    all_nums = normalized.split()

    if len(all_nums) == 7:
        # 单注：最后1个是蓝球，前面6个是红球
        nums_int = [int(n) for n in all_nums]
        blue = nums_int[-1]
        if 1 <= blue <= 16:
            reds = sorted(set(n for n in nums_int[:-1] if 1 <= n <= 33))
            if len(reds) >= 6:
                t = _build_ticket('ssq', reds[:6], [blue])
                if t:
                    tickets.append(t)

    elif len(all_nums) % 7 == 0 and len(all_nums) > 7:
        # 多注：每7个数字为一注
        count = len(all_nums) // 7
        for i in range(count):
            nums_int = [int(n) for n in all_nums[i*7:(i+1)*7]]
            blue = nums_int[-1]
            if 1 <= blue <= 16:
                reds = sorted(set(n for n in nums_int[:-1] if 1 <= n <= 33))
                if len(reds) >= 6:
                    t = _build_ticket('ssq', reds[:6], [blue])
                    if t and t not in tickets:
                        tickets.append(t)

    else:
        # 其他情况，尝试单注解析
        parsed = _parse_ssq_dash_format(line)
        if parsed:
            tickets.append(parsed)

    return tickets


def _parse_all_tickets(ocr_lines: List[Dict], lottery_type: Optional[str]) -> List[Dict]:
    """从所有 OCR 行中提取所有注号码，每行尝试解析一注，支持多注"""
    tickets = []

    for line_info in ocr_lines:
        text = line_info.get('text', '').strip()
        score = line_info.get('score', 0)

        if score < 0.3:
            continue
        if len(text) < 5:
            continue
        if re.search(r'第\s*\d{4,7}\s*期', text):
            continue

        # ── 优先：尝试 "红球-蓝球" 分隔格式（数字-数字）────────────
        # 这类格式在彩票图片中最常见，如 "02 06 10 16 27 28-06"
        multi_tickets = _parse_ssq_dash_format_multi(text)
        for mt in multi_tickets:
            t = _build_ticket(mt['type'], mt['reds'], mt['blues'])
            if t and t not in tickets:
                tickets.append(t)
                if lottery_type is None:
                    lottery_type = mt['type']
        if multi_tickets:
            continue

        # ── 其次：尝试 + | 红球/蓝球 分隔符 ─────────────────────────
        parsed = _parse_line_with_separator(text)
        if parsed:
            t = parsed['type']
            if lottery_type and t != lottery_type:
                pass
            else:
                ticket = _build_ticket(parsed['type'], parsed['reds'], parsed['blues'])
                if ticket and ticket not in tickets:
                    tickets.append(ticket)
                continue

        # ── 最后：无分隔符，提取所有数字尝试解析 ─────────────────────
        nums = _extract_numbers_from_line(text)
        if not nums:
            continue

        if lottery_type == 'ssq' or lottery_type is None:
            res = _try_parse_ssq_line(nums)
            if res:
                reds, blue = res
                ticket = _build_ticket('ssq', reds, [blue])
                if ticket and ticket not in tickets:
                    tickets.append(ticket)
                    if lottery_type is None:
                        lottery_type = 'ssq'
                    continue

        if lottery_type == 'dlt' or lottery_type is None:
            res = _try_parse_dlt_line(nums)
            if res:
                front, back = res
                ticket = _build_ticket('dlt', front, back)
                if ticket and ticket not in tickets:
                    tickets.append(ticket)
                    if lottery_type is None:
                        lottery_type = 'dlt'
                    continue

    return tickets


def _build_ticket(lottery_type: str, reds: List[int], blues: List[int]) -> Optional[Dict]:
    """构建标准化的 ticket 字典"""
    if lottery_type == 'ssq':
        if len(reds) < 6 or len(blues) < 1:
            return None
        return {
            'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
            'red4': reds[3], 'red5': reds[4], 'red6': reds[5],
            'blue': blues[0],
        }
    elif lottery_type == 'dlt':
        if len(reds) < 5 or len(blues) < 2:
            return None
        return {
            'red1': reds[0], 'red2': reds[1], 'red3': reds[2],
            'red4': reds[3], 'red5': reds[4],
            'blue': blues[0], 'blue2': blues[1],
        }
    return None


# ─────────────────────────────────────────────
# 数据库工具
# ─────────────────────────────────────────────

def _get_latest_issue(db_path: str) -> str:
    """获取数据库中最近一期期号"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT issue FROM ssq_history ORDER BY issue DESC LIMIT 1")
        ssq_issue = c.fetchone()
        c.execute("SELECT issue FROM dlt_history ORDER BY issue DESC LIMIT 1")
        dlt_issue = c.fetchone()
        conn.close()
        latest_issue = max(
            (int(ssq_issue[0]) if ssq_issue else 0),
            (int(dlt_issue[0]) if dlt_issue else 0)
        )
        return str(latest_issue + 1).zfill(5)
    except Exception:
        return "00001"


# ─────────────────────────────────────────────
# 主识别函数
# ─────────────────────────────────────────────

def recognize_lottery_image(image_path: str, db_path: str) -> Dict[str, Any]:
    """
    识别彩票图片（支持多注）
    使用 RapidOCR 引擎，纯 onnxruntime 实现，无需 torch

    Returns:
        {
            'success': bool,
            'lottery_type': str,        # 'ssq' | 'dlt'
            'issue': str,               # 期号
            'tickets': List[dict],      # 每注号码
            'error': str,
            'raw_text': str,            # OCR 原始文本（供调试）
            'raw_lines': List[dict],    # 逐行 OCR 结果
        }
    """
    result = {
        'success': False,
        'lottery_type': None,
        'issue': None,
        'tickets': [],
        'error': None,
        'raw_text': '',
        'raw_lines': [],
    }

    if not os.path.exists(image_path):
        result['error'] = f"图片文件不存在: {image_path}"
        return result

    try:
        # 加载图片
        try:
            from PIL import Image
            img = Image.open(image_path)
        except ImportError as e:
            result['error'] = f"缺少图像处理库: {e}\n请运行: pip install Pillow"
            return result
        except Exception as e:
            result['error'] = f"无法打开图片: {e}"
            return result

        # 图片预处理
        img_pil = _preprocess_image(img)

        # 获取 OCR 引擎
        ocr = get_ocr_engine()
        if ocr is None:
            error_msg = _ocr_init_error or "OCR 引擎未初始化"
            result['error'] = f"{error_msg}\n\n请先安装依赖：\npip install rapidocr_onnxruntime Pillow"
            return result

        # 将 PIL Image 转为 numpy array（RapidOCR 不接受 PIL Image）
        import numpy as np
        img_np = np.array(img_pil)

        # 执行 OCR（RapidOCR 自带多尺度检测，无需手动多次尝试）
        print(f"[OCR] 开始识别图片: {image_path}")
        try:
            raw_result, elapse = ocr(img_np, return_raw_result=True)
        except Exception as e:
            result['error'] = f"OCR 识别失败: {str(e)}"
            import traceback
            traceback.print_exc()
            return result

        # 将 RapidOCR 格式 [[box, text, score], ...] 转为 cnocr 兼容格式 [{text, score}, ...]
        ocr_lines = []
        if raw_result:
            for item in raw_result:
                if len(item) >= 3:
                    box, text, score = item[0], item[1], item[2]
                    try:
                        score_float = float(score)
                    except (ValueError, TypeError):
                        score_float = 0.5
                    ocr_lines.append({'text': str(text).strip(), 'score': score_float})
                    print(f"[OCR] 文本: {text}  (score={score_float:.2f})")

        if not ocr_lines:
            result['error'] = "未能从图片中识别出文字，请确保图片清晰可读"
            return result

        result['raw_lines'] = ocr_lines
        all_texts = [line['text'] for line in ocr_lines]
        raw_text = '\n'.join(all_texts)
        result['raw_text'] = raw_text

        if not raw_text.strip():
            result['error'] = "未识别到任何文字，请确保图片清晰"
            return result

        # 1. 提取期号
        issue = _extract_issue_from_text(raw_text)
        if issue is None:
            issue = _get_latest_issue(db_path)
            print(f"[OCR] 未识别到期号，使用默认值: {issue}")
        else:
            print(f"[OCR] 识别到期号: {issue}")
        result['issue'] = issue

        # 2. 判断彩票类型
        lottery_type = _classify_lottery_type(raw_text)
        print(f"[OCR] 识别到彩票类型: {lottery_type}")

        # 3. 逐行解析号码（支持多注）
        tickets = _parse_all_tickets(ocr_lines, lottery_type)

        # 如果逐行解析失败，尝试对全部数字整体解析
        if not tickets:
            all_nums = []
            for line in ocr_lines:
                all_nums.extend(_extract_numbers_from_line(line.get('text', '')))

            if all_nums:
                if lottery_type == 'ssq' or lottery_type is None:
                    res = _try_parse_ssq_line(all_nums)
                    if res:
                        reds, blue = res
                        t = _build_ticket('ssq', reds, [blue])
                        if t:
                            tickets.append(t)
                            if lottery_type is None:
                                lottery_type = 'ssq'

                if (lottery_type == 'dlt' or lottery_type is None) and not tickets:
                    res = _try_parse_dlt_line(all_nums)
                    if res:
                        front, back = res
                        t = _build_ticket('dlt', front, back)
                        if t:
                            tickets.append(t)
                            if lottery_type is None:
                                lottery_type = 'dlt'

        if not tickets:
            result['error'] = (
                f"未能识别出有效号码\n\n"
                f"OCR 原始文本：\n{raw_text}\n\n"
                f"提示：请确保图片清晰，号码区域完整可见"
            )
            return result

        # 推断最终彩票类型（从 tickets 判断）
        if lottery_type is None:
            if 'blue2' in tickets[0]:
                lottery_type = 'dlt'
            else:
                lottery_type = 'ssq'

        result['lottery_type'] = lottery_type
        result['tickets'] = tickets
        result['success'] = True

        print(f"[OCR] 识别完成: 类型={lottery_type}, 期号={issue}, 共{len(tickets)}注")
        for i, t in enumerate(tickets, 1):
            print(f"[OCR]   第{i}注: {t}")

    except Exception as e:
        result['error'] = f"OCR识别异常: {str(e)}"
        import traceback
        traceback.print_exc()

    return result


def recognize_lottery_image_async(image_path: str, db_path: str, callback,
                                   progress_callback=None):
    """异步识别彩票图片"""
    def _do_recognize():
        if progress_callback:
            progress_callback("正在预处理图片...")
        result = recognize_lottery_image(image_path, db_path)
        if callback:
            callback(result)

    thread = threading.Thread(target=_do_recognize, daemon=True)
    thread.start()
