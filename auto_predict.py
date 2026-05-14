# -*- coding: utf-8 -*-
"""
自动拉取历史数据并输出预测结果（命令行版本，无GUI）
支持双色球（SSQ）和大乐透（DLT）两种彩票
支持12种预测方案（🚀 优化版：并行预测）

🚀 优化：使用 concurrent.futures 并行执行多个预测方案
"""
import os
import sys
import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# 修复Windows控制台编码问题
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置路径（支持打包后运行）
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BASE_DIR

# 数据库使用 APP_DIR（用户数据目录）
DB_PATH = os.path.join(APP_DIR, "ssq.db")

sys.path.insert(0, BASE_DIR)
import fetch_data
import fetch_dlt
import predict
import predict_dlt


def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_with_progress(label, db_path, method, module):
    """运行单个预测，打印进度。"""
    print(f"  -> {label}...", end="", flush=True)
    result = module.predict_next_issue(db_path, method=method)
    if result and "error" not in result:
        print("  [OK]")
    else:
        err = result.get("error", "未知错误") if result else "返回空结果"
        print(f"  [FAIL] ({err[:40]})")
    return result




def run_parallel_predict(db_path, methods, module, max_workers=4):
    """
    🚀 优化版：并行执行多个预测方案

    Args:
        db_path: 数据库路径
        methods: [(method_key, method_name), ...]
        module: predict 或 predict_dlt 模块
        max_workers: 最大并行数

    Returns:
        dict: {method_key: result, ...}
    """
    results = {}

    def predict_one(method_key, label):
        try:
            result = module.predict_next_issue(db_path, method=method_key)
            return method_key, label, result, None
        except Exception as e:
            return method_key, label, None, str(e)

    print(f"  [Parallel] Starting {len(methods)} predictions with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(predict_one, method_key, label): (method_key, label)
            for method_key, label in methods
        }

        completed = 0
        for future in as_completed(futures):
            method_key, label, result, error = future.result()
            completed += 1

            if error:
                print(f"  [{completed}/{len(methods)}] {label}: [FAIL] {error[:40]}")
                results[method_key] = {"error": error}
            elif result and "error" not in result:
                print(f"  [{completed}/{len(methods)}] {label}: [OK]")
                results[method_key] = result
            else:
                err = result.get("error", "Unknown") if result else "Empty"
                print(f"  [{completed}/{len(methods)}] {label}: [FAIL] {err[:40]}")
                results[method_key] = result or {"error": err}

    return results



def build_report_ssq(results, methods, W=64):
    """构建双色球预测报告内容"""
    base = None
    for method, _ in methods:
        r = results.get(method)
        if r and "error" not in r:
            base = r
            break

    next_issue   = base.get("next_issue",   "未知") if base else "未知"
    history_used = base.get("history_used", "?")    if base else "?"

    lines = []
    lines.append("=" * W)
    lines.append(f"  双色球  第 {next_issue} 期  预测报告")
    lines.append(f"  基于历史 {history_used} 期数据  |  生成时间: {get_current_time()}")
    lines.append("=" * W)
    lines.append("")
    lines.append("-" * W)
    for _, name in methods:
        lines.append(name)
    lines.append("")
    for method, _ in methods:
        r = results.get(method)
        if r and "error" not in r:
            reds = " ".join([f"{n:02d}" for n in r.get("red_balls", [])])
            blue = r.get("blue_ball", "")
            blue_str = f"{blue:02d}" if isinstance(blue, int) else str(blue)
            lines.append(f"{reds} + {blue_str}")
        else:
            err = r.get("error", "预测失败") if r else "数据不足"
            lines.append(f"[X] {err[:30]}")
    lines.append("")
    lines.append("=" * W)
    lines.append("  [!] 以上预测仅供娱乐参考，彩票开奖结果完全随机，请理性购彩。")
    lines.append("=" * W)
    return "\n".join(lines), next_issue


def build_report_dlt(results, methods, W=64):
    """构建大乐透预测报告内容（蓝球为2个，用 | 分隔前后区）"""
    base = None
    for method, _ in methods:
        r = results.get(method)
        if r and "error" not in r:
            base = r
            break

    next_issue   = base.get("next_issue",   "未知") if base else "未知"
    history_used = base.get("history_used", "?")    if base else "?"

    lines = []
    lines.append("=" * W)
    lines.append(f"  大乐透  第 {next_issue} 期  预测报告")
    lines.append(f"  基于历史 {history_used} 期数据  |  生成时间: {get_current_time()}")
    lines.append("=" * W)
    lines.append("")
    lines.append("-" * W)
    for _, name in methods:
        lines.append(name)
    lines.append("")
    for method, _ in methods:
        r = results.get(method)
        if r and "error" not in r:
            reds = " ".join([f"{n:02d}" for n in r.get("red_balls", [])])
            blues = r.get("blue_balls", [])
            if blues:
                blue_str = " ".join([f"{b:02d}" for b in blues])
            else:
                blue = r.get("blue_ball", "")
                blue_str = f"{blue:02d}" if isinstance(blue, int) else str(blue)
            lines.append(f"{reds} | {blue_str}")
        else:
            err = r.get("error", "预测失败") if r else "数据不足"
            lines.append(f"[X] {err[:30]}")
    lines.append("")
    lines.append("=" * W)
    lines.append("  [!] 以上预测仅供娱乐参考，彩票开奖结果完全随机，请理性购彩。")
    lines.append("=" * W)
    return "\n".join(lines), next_issue


def main():
    W = 64
    print("=" * W)
    print("  彩票自动预测工具  v3.0  (双色球 + 大乐透)")
    print("=" * W)
    print()

    # ── 1. 初始化数据库 ──────────────────────────────────────
    fetch_data.init_db(DB_PATH)
    fetch_dlt.init_dlt_db(DB_PATH)

    # ── 2. 拉取历史数据 ──────────────────────────────────────
    print("【1/5】正在拉取双色球历史数据...")
    print("-" * W)
    count_ssq = fetch_data.fetch_all(DB_PATH, progress_cb=print)
    print(f"[OK] 双色球数据更新完成，新增 {count_ssq} 条记录")
    print()

    print("【2/5】正在拉取大乐透历史数据...")
    print("-" * W)
    count_dlt = fetch_dlt.fetch_all(DB_PATH, progress_cb=print)
    print(f"[OK] 大乐透数据更新完成，新增 {count_dlt} 条记录")
    print()

    _METHODS = [
        ("linear",        "线性回归"),
        ("entropy",       "熵最大化"),
        ("lstm",          "MLP神经网络"),
        ("xgboost",       "XGBoost"),
        ("gru",           "GRU序列"),
        ("hot_cold",      "冷热号平衡"),
        ("same_period",   "同期历史"),
        ("consecutive",   "连号跨度"),
        ("odd_even",      "奇偶比"),
        ("random_forest", "随机森林"),
        ("bayesian",      "贝叶斯"),
        ("genetic",       "遗传算法"),
    ]

    # ── 3. 双色球预测（🚀 并行优化）───────────────────────
    print("【3/5】正在执行双色球预测（共12种方案）...")
    print("-" * W)
    results_ssq = run_parallel_predict(DB_PATH, _METHODS, predict, max_workers=4)
    print()

    # ── 4. 大乐透预测（🚀 并行优化）───────────────────────
    print("【4/5】正在执行大乐透预测（共12种方案）...")
    print("-" * W)
    results_dlt = run_parallel_predict(DB_PATH, _METHODS, predict_dlt, max_workers=4)
    print()

    # ── 5. 构建 & 保存文件 ──────────────────────────────────
    print("【5/5】正在保存预测报告...")
    pred_dir = os.path.join(BASE_DIR, "预测结果")
    os.makedirs(pred_dir, exist_ok=True)

    # 双色球报告
    content_ssq, next_ssq = build_report_ssq(results_ssq, _METHODS, W)
    path_ssq = os.path.join(pred_dir, f"SSQ预测_{next_ssq}.txt")
    with open(path_ssq, "w", encoding="utf-8") as f:
        f.write(content_ssq)

    # 大乐透报告
    content_dlt, next_dlt = build_report_dlt(results_dlt, _METHODS, W)
    path_dlt = os.path.join(pred_dir, f"DLT预测_{next_dlt}.txt")
    with open(path_dlt, "w", encoding="utf-8") as f:
        f.write(content_dlt)

    # ── 6. 控制台输出 ─────────────────────────────────────────
    print()
    print(content_ssq)
    print()
    print(content_dlt)
    print()
    print(f"[OK] 双色球报告已保存至: {path_ssq}")
    print(f"[OK] 大乐透报告已保存至: {path_dlt}")
    print("=" * W)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)
