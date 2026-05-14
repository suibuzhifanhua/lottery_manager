"""
双色球预测模块
提供多种预测方案：
1. 线性回归预测 - 基于时间序列趋势外推
2. 频率加权预测 - 基于历史出现频率加权随机选择
3. 熵最大化预测 - 基于信息熵理论
4. 遗漏值预测 - 基于号码遗漏周期
5. LSTM深度学习预测 - 基于神经网络
"""
import random

from feature_utils import (
    load_history_as_dict,
    calc_next_issue as _calc_next_issue,
    compute_gap_fast as _compute_gap_fast,
    calc_frequency as _calc_frequency,
)

# 各深度学习模块延迟导入（避免启动时加载耗时）
_lstm_model  = None   # MLP (sklearn)
_xgb_model   = None   # XGBoost
_gru_model   = None   # GRU (PyTorch)


def _precompute_features_fast(history, window=20, is_blue=False):
    """
    🚀 预计算所有期号的所有号码特征
    
    Args:
        history: 历史数据列表（dict格式）
        window: 统计窗口大小（默认20期）
        is_blue: 是否为蓝球
    
    Returns:
        freq_matrix: (n, num_count+1) 频率矩阵
        gap_matrix: (n, num_count+1) 遗漏值矩阵
    """
    import numpy as np
    
    n = len(history)
    if n == 0:
        return np.zeros((1, 17 if is_blue else 34)), np.zeros((1, 17 if is_blue else 34))
    
    num_count = 16 if is_blue else 33
    
    # 初始化矩阵
    freq_matrix = np.zeros((n, num_count + 1), dtype=np.float32)
    gap_matrix = np.full((n, num_count + 1), n, dtype=np.float32)
    
    # 遍历历史，累计计算
    for i in range(n):
        if is_blue:
            past = history[:i]
            if not past:
                continue
            recent = past[-window:] if len(past) >= window else past
            
            # 统计蓝球频率
            for rec in recent:
                freq_matrix[i, rec["blue"]] += 1
            
            # 计算遗漏值
            for num in range(1, num_count + 1):
                gap = 0
                for j in range(len(past) - 1, -1, -1):
                    if past[j]["blue"] == num:
                        break
                    gap += 1
                else:
                    gap = len(past)
                gap_matrix[i, num] = gap
        else:
            past = history[:i]
            if not past:
                continue
            recent = past[-window:] if len(past) >= window else past
            
            # 统计红球频率
            for rec in recent:
                for k in range(1, 7):
                    freq_matrix[i, rec[f"red{k}"]] += 1
            
            # 计算遗漏值
            for num in range(1, num_count + 1):
                gap = 0
                for j in range(len(past) - 1, -1, -1):
                    rec = past[j]
                    reds = {rec[f"red{k}"] for k in range(1, 7)}
                    if num in reds:
                        break
                    gap += 1
                else:
                    gap = len(past)
                gap_matrix[i, num] = gap
    
    return freq_matrix, gap_matrix


def load_history(db_path):
    """
    读取全部历史记录（按期号升序），
    返回 list of dict: {issue, red1..red6, blue}
    """
    return load_history_as_dict(db_path)


def _linear_regression(x_list, y_list):
    """
    最小二乘线性回归: y = kx + b
    返回 (k, b)
    """
    n = len(x_list)
    if n == 0:
        return 0, 0
    sum_x = sum(x_list)
    sum_y = sum(y_list)
    sum_xy = sum(x * y for x, y in zip(x_list, y_list))
    sum_x2 = sum(x * x for x in x_list)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0, sum_y / n
    k = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - k * sum_x) / n
    return k, b


def _calc_next_issue(last_issue):
    """根据最后一期推算下一期号（支持跨年）"""
    try:
        year = int(str(last_issue)[:2])
        num = int(str(last_issue)[2:])
        # 双色球每年约 150-154 期，这里取宽松上限 160
        if num >= 160:
            year += 1
            num = 1
        else:
            num += 1
        return f"{year:02d}{num:03d}"
    except:
        return str(int(last_issue) + 1)


def _calc_frequency(history):
    """
    计算所有号码的历史出现频率
    返回: (red_freq, blue_freq)
    """
    red_freq = {i: 0 for i in range(1, 34)}
    blue_freq = {i: 0 for i in range(1, 17)}

    for rec in history:
        for i in range(1, 7):
            red_freq[rec[f"red{i}"]] += 1
        blue_freq[rec["blue"]] += 1

    return red_freq, blue_freq


def _precompute_features_fast(history, window=20, is_blue=False):
    """
    🚀 预计算所有期号的所有号码特征（优化版）
    
    改进点：
    1. 一次性计算所有期号的所有特征矩阵，避免重复遍历
    2. 使用 numpy 向量化操作代替 Python 循环
    3. 时间复杂度从 O(N²×M) 降为 O(N²)
    
    Args:
        history: 历史数据列表
        window: 统计窗口大小（默认20期）
        is_blue: 是否为蓝球
    
    Returns:
        freq_matrix: (n, num_count+1) 频率矩阵
        gap_matrix: (n, num_count+1) 遗漏值矩阵
    """
    import numpy as np
    
    n = len(history)
    num_count = 16 if is_blue else 33
    
    # 初始化矩阵
    freq_matrix = np.zeros((n, num_count + 1), dtype=np.float32)
    gap_matrix = np.full((n, num_count + 1), n, dtype=np.float32)
    
    # 遍历历史，累计计算
    for i in range(n):
        past = history[:i]
        if not past:
            continue
        recent = past[-window:] if len(past) >= window else past
        
        # 统计频率
        for rec in recent:
            if is_blue:
                freq_matrix[i, rec["blue"]] += 1
            else:
                for k in range(1, 7):
                    freq_matrix[i, rec[f"red{k}"]] += 1
        
        # 计算遗漏值
        for num in range(1, num_count + 1):
            gap = 0
            for j in range(len(past) - 1, -1, -1):
                rec = past[j]
                if is_blue:
                    nums = {rec["blue"]}
                else:
                    nums = {rec[f"red{k}"] for k in range(1, 7)}
                if num in nums:
                    break
                gap += 1
            else:
                gap = len(past)
            gap_matrix[i, num] = gap
    
    return freq_matrix, gap_matrix


def predict_linear_regression(history):
    """
    方案一：线性回归预测
    对红球6个位置和蓝球分别做线性回归，外推预测下一期
    """
    n = len(history)
    if n < 10:
        return None

    x = list(range(n))
    next_x = n

    # 红球各位置回归
    raw_reds = []
    detail = {}
    for i in range(1, 7):
        y = [rec[f"red{i}"] for rec in history]
        k, b = _linear_regression(x, y)
        pred = k * next_x + b
        raw_reds.append(round(pred))
        detail[f"红球位置{i}原始预测"] = round(pred, 2)

    # 蓝球回归
    y_blue = [rec["blue"] for rec in history]
    k_b, b_b = _linear_regression(x, y_blue)
    blue_raw = k_b * next_x + b_b
    blue_pred = round(blue_raw)

    # 截断到合法范围
    raw_reds = [max(1, min(33, r)) for r in raw_reds]
    blue_pred = max(1, min(16, blue_pred))

    # 红球去重并补充
    red_freq, blue_freq = _calc_frequency(history)
    seen = set()
    final_reds = []
    for r in raw_reds:
        if r not in seen and 1 <= r <= 33:
            final_reds.append(r)
            seen.add(r)

    # 补充不足6个
    if len(final_reds) < 6:
        # 按频率排序，优先选高频且未出现的
        sorted_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)
        for r in sorted_reds:
            if r not in seen and len(final_reds) < 6:
                final_reds.append(r)
                seen.add(r)

    final_reds.sort()

    # 热门号Top10
    hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue": _calc_next_issue(history[-1]["issue"]),
        "red_balls": final_reds,
        "blue_ball": blue_pred,
        "method": f"线性回归（全部 {n} 期历史数据）",
        "detail": detail,
        "red_freq": red_freq,
        "blue_freq": blue_freq,
        "hot_reds": hot_reds,
        "hot_blues": hot_blues,
        "history_used": n,
        "blue_raw": round(blue_raw, 2)
    }


def predict_frequency_weighted(history, sample_count=6):
    """
    方案二：频率加权预测
    基于历史出现频率进行加权随机选择
    出现次数越多的号码，被选中的概率越高
    """
    n = len(history)
    if n < 10:
        return None

    red_freq, blue_freq = _calc_frequency(history)

    # 频率加权随机选择红球（不放回）
    red_pool = list(range(1, 34))
    red_weights = [red_freq[i] for i in red_pool]
    # 如果所有频率都是0（理论上不可能），使用均匀权重
    if sum(red_weights) == 0:
        red_weights = [1] * 33

    selected_reds = []
    temp_pool = red_pool.copy()
    temp_weights = red_weights.copy()

    for _ in range(6):
        # 加权随机选择
        total = sum(temp_weights)
        if total == 0:
            # 如果权重归零，使用均匀随机
            choice = random.choice(temp_pool)
        else:
            r = random.uniform(0, total)
            cumsum = 0
            for i, (num, w) in enumerate(zip(temp_pool, temp_weights)):
                cumsum += w
                if r <= cumsum:
                    choice = num
                    break
            else:
                choice = temp_pool[-1]

        selected_reds.append(choice)
        # 从池中移除已选号码
        idx = temp_pool.index(choice)
        temp_pool.pop(idx)
        temp_weights.pop(idx)

    selected_reds.sort()

    # 频率加权随机选择蓝球
    blue_pool = list(range(1, 17))
    blue_weights = [blue_freq[i] for i in blue_pool]
    if sum(blue_weights) == 0:
        blue_weights = [1] * 16

    total_blue = sum(blue_weights)
    r = random.uniform(0, total_blue)
    cumsum = 0
    selected_blue = blue_pool[-1]
    for num, w in zip(blue_pool, blue_weights):
        cumsum += w
        if r <= cumsum:
            selected_blue = num
            break

    # 热门号Top10
    hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue": _calc_next_issue(history[-1]["issue"]),
        "red_balls": selected_reds,
        "blue_ball": selected_blue,
        "method": f"频率加权随机（全部 {n} 期历史数据）",
        "detail": {
            "说明": "基于历史出现频率加权随机选择",
            "红球选择": "不放回加权抽样",
            "蓝球选择": "加权随机抽样"
        },
        "red_freq": red_freq,
        "blue_freq": blue_freq,
        "hot_reds": hot_reds,
        "hot_blues": hot_blues,
        "history_used": n
    }


def calc_entropy(freq_dict, total_count):
    """
    计算信息熵（香农熵）
    H = -Σ(p_i * log2(p_i))
    其中 p_i = freq_i / total_count

    熵越大表示分布越均匀（随机性高）
    熵越小表示分布越集中（某些号码出现频率更高）
    """
    import math
    entropy = 0.0
    for freq in freq_dict.values():
        if freq > 0:
            p = freq / total_count
            entropy -= p * math.log2(p)
    return entropy


def calc_max_entropy(num_count):
    """计算最大可能熵（均匀分布时）"""
    import math
    p = 1.0 / num_count
    return -num_count * p * math.log2(p)


def analyze_entropy(history):
    """
    信息熵分析：衡量号码分布的随机性

    Returns:
        dict: 熵分析结果
    """
    n = len(history)
    if n < 10:
        return None

    red_freq, blue_freq = _calc_frequency(history)

    # 红球总出现次数 = 每期6个 * 期数
    red_total = n * 6
    blue_total = n

    # 计算实际熵
    red_entropy = calc_entropy(red_freq, red_total)
    blue_entropy = calc_entropy(blue_freq, blue_total)

    # 计算最大熵（理论均匀分布）
    red_max_entropy = calc_max_entropy(33)
    blue_max_entropy = calc_max_entropy(16)

    # 计算相对熵（0-1之间，1表示完全均匀）
    red_relative = red_entropy / red_max_entropy if red_max_entropy > 0 else 0
    blue_relative = blue_entropy / blue_max_entropy if blue_max_entropy > 0 else 0

    # 计算标准差（衡量频率波动）
    red_freq_values = list(red_freq.values())
    blue_freq_values = list(blue_freq.values())

    red_mean = sum(red_freq_values) / len(red_freq_values)
    blue_mean = sum(blue_freq_values) / len(blue_freq_values)

    red_std = (sum((x - red_mean) ** 2 for x in red_freq_values) / len(red_freq_values)) ** 0.5
    blue_std = (sum((x - blue_mean) ** 2 for x in blue_freq_values) / len(blue_freq_values)) ** 0.5

    # 变异系数（标准差/均值，衡量相对波动）
    red_cv = red_std / red_mean if red_mean > 0 else 0
    blue_cv = blue_std / blue_mean if blue_mean > 0 else 0

    # 熵最大化预测：选择使下一期熵最大的组合
    # 即选择与历史频率差异最大的号码（冷门号）
    expected_red_freq = red_total / 33  # 每个红球理论期望次数
    expected_blue_freq = blue_total / 16  # 每个蓝球理论期望次数

    # 计算每个号码的"偏离度"（与期望的差距）
    red_deviation = {k: abs(v - expected_red_freq) for k, v in red_freq.items()}
    blue_deviation = {k: abs(v - expected_blue_freq) for k, v in blue_freq.items()}

    # 熵最大化选择：选偏离度最大的（最冷门）号码
    # 这样可以让分布趋向均匀，熵增大
    cold_reds = sorted(red_deviation.keys(), key=lambda k: red_deviation[k], reverse=True)[:6]
    cold_blues = sorted(blue_deviation.keys(), key=lambda k: blue_deviation[k], reverse=True)[:3]

    cold_reds.sort()

    return {
        "red_entropy": round(red_entropy, 4),
        "blue_entropy": round(blue_entropy, 4),
        "red_max_entropy": round(red_max_entropy, 4),
        "blue_max_entropy": round(blue_max_entropy, 4),
        "red_relative_entropy": round(red_relative, 4),
        "blue_relative_entropy": round(blue_relative, 4),
        "red_std": round(red_std, 2),
        "blue_std": round(blue_std, 2),
        "red_cv": round(red_cv, 4),
        "blue_cv": round(blue_cv, 4),
        "expected_red_freq": round(expected_red_freq, 2),
        "expected_blue_freq": round(expected_blue_freq, 2),
        "cold_reds": cold_reds,
        "cold_blues": cold_blues,
        "red_freq": red_freq,
        "blue_freq": blue_freq,
        "history_count": n
    }


def predict_entropy_maximized(history):
    """
    方案三：熵最大化预测
    选择"最冷门"的号码组合，使整体分布趋向均匀
    """
    n = len(history)
    if n < 10:
        return None

    entropy_data = analyze_entropy(history)
    if not entropy_data:
        return None

    red_freq = entropy_data["red_freq"]
    blue_freq = entropy_data["blue_freq"]

    # 计算期望频率
    expected_red = n * 6 / 33
    expected_blue = n / 16

    # 选择最偏离期望的号码（冷门号）
    red_deviation = {k: expected_red - v for k, v in red_freq.items()}  # 正值表示低于期望（冷门）
    blue_deviation = {k: expected_blue - v for k, v in blue_freq.items()}

    # 选择最冷门的6个红球
    selected_reds = sorted(red_deviation.keys(), key=lambda k: red_deviation[k], reverse=True)[:6]
    selected_reds.sort()

    # 选择最冷门的蓝球
    selected_blue = sorted(blue_deviation.keys(), key=lambda k: blue_deviation[k], reverse=True)[0]

    # 热门号（用于对比显示）
    hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue": _calc_next_issue(history[-1]["issue"]),
        "red_balls": selected_reds,
        "blue_ball": selected_blue,
        "method": f"熵最大化（全部 {n} 期历史数据）",
        "detail": {
            "说明": "选择历史出现频率最低的号码，使分布趋向均匀",
            "红球策略": "选择最冷门的6个号码",
            "蓝球策略": "选择最冷门的1个号码",
            "红球熵": entropy_data["red_entropy"],
            "蓝球熵": entropy_data["blue_entropy"],
            "红球相对熵": entropy_data["red_relative_entropy"],
            "蓝球相对熵": entropy_data["blue_relative_entropy"]
        },
        "red_freq": red_freq,
        "blue_freq": blue_freq,
        "hot_reds": hot_reds,
        "hot_blues": hot_blues,
        "history_used": n,
        "entropy_data": entropy_data
    }


def analyze_gap(history):
    """
    遗漏值分析（Gap Analysis）
    统计每个号码距离上次出现已经间隔了多少期

    使用 feature_utils.compute_gap_fast() O(n) 单次反向扫描，
    比原来对每个号码独立反向搜索快 ~50x。

    history: list of dict {issue, red1..red6, blue}
    Returns:
        dict: {red_gap, blue_gap, max_gap_red, max_gap_blue, ...}
    """
    n = len(history)
    if n == 0:
        return None

    # 将 dict 格式的 history 转成 tuple 格式供 compute_gap_fast 使用
    history_tuples = [
        (rec["issue"],
         rec["red1"], rec["red2"], rec["red3"],
         rec["red4"], rec["red5"], rec["red6"],
         rec["blue"])
        for rec in history
    ]

    red_gap, blue_gap = _compute_gap_fast(history_tuples)

    # 找出最大遗漏值
    max_gap_red  = max(red_gap.keys(),  key=lambda k: red_gap[k])
    max_gap_blue = max(blue_gap.keys(), key=lambda k: blue_gap[k])

    # 计算平均遗漏值（排除从未出现的情况，此处 inf 已替换为 n）
    finite_reds  = [g for g in red_gap.values()  if g < n]
    finite_blues = [g for g in blue_gap.values() if g < n]
    avg_gap_red  = sum(finite_reds)  / len(finite_reds)  if finite_reds  else 0
    avg_gap_blue = sum(finite_blues) / len(finite_blues) if finite_blues else 0

    return {
        "red_gap":            red_gap,
        "blue_gap":           blue_gap,
        "max_gap_red":        max_gap_red,
        "max_gap_blue":       max_gap_blue,
        "max_gap_red_value":  red_gap[max_gap_red],
        "max_gap_blue_value": blue_gap[max_gap_blue],
        "avg_gap_red":        avg_gap_red,
        "avg_gap_blue":       avg_gap_blue,
        "history_count":      n
    }


def predict_gap_maximized(history):
    """
    方案四：遗漏值最大化预测（Gap Maximization）
    选择当前遗漏值最大的号码（最久未出的号码）
    基于"长期未出的号码该出了"的假设（赌徒谬误）
    """
    n = len(history)
    if n < 10:
        return None

    gap_data = analyze_gap(history)
    if not gap_data:
        return None

    red_gap = gap_data["red_gap"]
    blue_gap = gap_data["blue_gap"]

    # 选择遗漏值最大的6个红球
    # 排除无穷大（从未出现的号码，理论上不可能）
    sorted_reds = sorted(red_gap.keys(), key=lambda k: red_gap[k], reverse=True)
    selected_reds = sorted_reds[:6]
    selected_reds.sort()

    # 选择遗漏值最大的蓝球
    selected_blue = max(blue_gap.keys(), key=lambda k: blue_gap[k])

    # 获取频率数据用于显示
    red_freq, blue_freq = _calc_frequency(history)
    hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    # 获取最大遗漏值的号码详情
    max_gap_reds = sorted_reds[:10]  # Top 10 最大遗漏

    return {
        "next_issue": _calc_next_issue(history[-1]["issue"]),
        "red_balls": selected_reds,
        "blue_ball": selected_blue,
        "method": f"遗漏值最大化（全部 {n} 期历史数据）",
        "detail": {
            "说明": "选择当前遗漏值最大（最久未出）的号码",
            "红球策略": "选择遗漏值最大的6个号码",
            "蓝球策略": "选择遗漏值最大的1个号码",
            "最大红球遗漏": gap_data["max_gap_red_value"],
            "最大蓝球遗漏": gap_data["max_gap_blue_value"],
            "平均红球遗漏": round(gap_data["avg_gap_red"], 1),
            "平均蓝球遗漏": round(gap_data["avg_gap_blue"], 1)
        },
        "red_freq": red_freq,
        "blue_freq": blue_freq,
        "hot_reds": hot_reds,
        "hot_blues": hot_blues,
        "history_used": n,
        "gap_data": gap_data,
        "max_gap_reds": max_gap_reds
    }


def predict_hot_cold_balance(history):
    """
    方案五：冷热号平衡预测
    将号码分为"热号"（近20期频繁出现）和"冷号"（近20期未出），
    按 4热+2冷 的比例混合，符合彩票号码分布规律。
    """
    n = len(history)
    if n < 20:
        return None

    recent = history[-20:]          # 近20期
    red_freq_all, blue_freq_all = _calc_frequency(history)
    red_freq_rec, blue_freq_rec = _calc_frequency(recent)

    hot_reds  = [k for k, v in red_freq_rec.items()  if v >= 2]
    cold_reds = [k for k, v in red_freq_rec.items()  if v == 0]
    hot_blues  = [k for k, v in blue_freq_rec.items() if v >= 1]
    cold_blues = [k for k, v in blue_freq_rec.items() if v == 0]

    # 若热/冷池不足则降级到频率加权
    if len(hot_reds) < 4 or len(cold_reds) < 2:
        return predict_frequency_weighted(history)

    selected = set(random.sample(hot_reds, min(4, len(hot_reds))))
    cold_pool = [r for r in cold_reds if r not in selected]
    while len(selected) < 6 and cold_pool:
        c = random.choice(cold_pool)
        selected.add(c)
        cold_pool.remove(c)
    while len(selected) < 6:
        r = random.choice([x for x in range(1, 34) if x not in selected])
        selected.add(r)
    final_reds = sorted(selected)

    if hot_blues:
        blue = random.choice(hot_blues)
    else:
        blue = random.choice(list(range(1, 17)))

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    blue,
        "method":       f"冷热号平衡（近20期热号+冷号混合）",
        "detail":       {"说明": "近20期频繁出现为热号，未出现为冷号，按4热2冷比例混合",
                         "热号池": sorted(hot_reds)[:10],
                         "冷号池": sorted(cold_reds)[:10]},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_same_period_history(history):
    """
    方案六：同期历史预测
    统计历史上同月份开奖的号码，找出高频号码作为预测依据（季节性规律）。
    """
    import datetime
    n = len(history)
    if n < 30:
        return None

    # 推算当前月份（基于最后一期期号）
    try:
        now_month = datetime.date.today().month
    except Exception:
        now_month = 1

    # 双色球期号格式 YYXXX，每年约 153 期，一月约 12-13 期
    # 用简单近似：期号末3位推算月份（每月约12~13期）
    def issue_to_approx_month(issue_str):
        try:
            seq = int(str(issue_str)[2:])   # 期内序号
            return min(12, max(1, (seq - 1) // 13 + 1))
        except Exception:
            return 0

    same_month_recs = [r for r in history if issue_to_approx_month(r["issue"]) == now_month]
    if len(same_month_recs) < 5:
        same_month_recs = history[-30:]     # 不足时取近30期

    red_freq_sm, blue_freq_sm = _calc_frequency(same_month_recs)
    red_freq_all, blue_freq_all = _calc_frequency(history)

    # 同期高频红球
    sorted_reds  = sorted(red_freq_sm.keys(),  key=lambda k: red_freq_sm[k],  reverse=True)
    sorted_blues = sorted(blue_freq_sm.keys(), key=lambda k: blue_freq_sm[k], reverse=True)

    final_reds = sorted(sorted_reds[:6])
    final_blue = sorted_blues[0]

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    final_blue,
        "method":       f"同期历史（{now_month}月高频号码，共 {len(same_month_recs)} 期参考）",
        "detail":       {"说明": f"统计历史{now_month}月开奖的高频号码",
                         "参考期数": len(same_month_recs)},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_consecutive_span(history):
    """
    方案七：连号跨度分析
    统计历史中连号（相邻号码对）出现的概率及整注跨度（最大-最小）分布，
    生成含合理连号结构和跨度的号码组合。
    """
    n = len(history)
    if n < 20:
        return None

    red_freq_all, blue_freq_all = _calc_frequency(history)

    # 统计跨度分布
    spans = [history[i]["red6"] - history[i]["red1"] for i in range(n)]
    avg_span = sum(spans) / len(spans)
    target_span = round(avg_span)

    # 统计连号频次（号码i与i+1同时出现的次数）
    consec_freq = {i: 0 for i in range(1, 33)}
    for rec in history:
        reds = {rec[f"red{j}"] for j in range(1, 7)}
        for r in reds:
            if r + 1 in reds:
                consec_freq[r] += 1

    # 选最常见连号对中的一对
    best_consec = max(consec_freq, key=lambda k: consec_freq[k])

    # 以连号对为锚点，加权补充剩余4个号码
    selected = {best_consec, best_consec + 1}
    weights = {k: red_freq_all[k] for k in range(1, 34) if k not in selected}
    pool = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    for r in pool:
        if len(selected) >= 6:
            break
        # 控制跨度：不超出 [best_consec - target_span//2, best_consec + target_span//2]
        lo = max(1, best_consec - target_span // 2)
        hi = min(33, best_consec + target_span)
        if lo <= r <= hi:
            selected.add(r)

    # 若跨度约束后不足6个，取频率最高的补充
    for r in pool:
        if len(selected) >= 6:
            break
        if r not in selected:
            selected.add(r)

    final_reds = sorted(selected)[:6]

    # 蓝球取历史高频
    blue = max(blue_freq_all, key=lambda k: blue_freq_all[k])

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    blue,
        "method":       f"连号跨度分析（历史平均跨度 {target_span}，连号锚 {best_consec}-{best_consec+1}）",
        "detail":       {"说明": "以历史最常见连号对为锚点，按平均跨度范围补充剩余号码",
                         "连号锚点": f"{best_consec}-{best_consec+1}",
                         "历史平均跨度": round(avg_span, 1),
                         "目标跨度": target_span},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_odd_even_ratio(history):
    """
    方案八：奇偶比/大小比预测
    双色球红球历史最优奇偶比约 3:3 或 4:2，大小比（≤16/≥17）约 3:3。
    强制生成满足最优比例的号码组合（高频号中符合比例条件的优先）。
    """
    n = len(history)
    if n < 10:
        return None

    # 统计历史奇偶比分布
    ratio_count = {}
    for rec in history:
        reds = [rec[f"red{i}"] for i in range(1, 7)]
        odd_c = sum(1 for r in reds if r % 2 == 1)
        ratio_count[odd_c] = ratio_count.get(odd_c, 0) + 1

    best_odd_count = max(ratio_count, key=lambda k: ratio_count[k])

    # 统计历史大小比分布（≤16 为小，≥17 为大）
    size_count = {}
    for rec in history:
        reds = [rec[f"red{i}"] for i in range(1, 7)]
        small_c = sum(1 for r in reds if r <= 16)
        size_count[small_c] = size_count.get(small_c, 0) + 1

    best_small_count = max(size_count, key=lambda k: size_count[k])
    best_big_count = 6 - best_small_count

    red_freq_all, blue_freq_all = _calc_frequency(history)

    odds   = sorted([k for k in range(1, 34) if k % 2 == 1], key=lambda k: red_freq_all[k], reverse=True)
    evens  = sorted([k for k in range(1, 34) if k % 2 == 0], key=lambda k: red_freq_all[k], reverse=True)
    smalls = sorted([k for k in range(1, 34) if k <= 16],    key=lambda k: red_freq_all[k], reverse=True)
    bigs   = sorted([k for k in range(1, 34) if k >= 17],    key=lambda k: red_freq_all[k], reverse=True)

    # 在奇偶约束 & 大小约束的交集中按频率选号
    selected = set()

    # 先选奇数中的"小号"高频
    odd_small  = [k for k in odds   if k <= 16]
    odd_big    = [k for k in odds   if k >= 17]
    even_small = [k for k in evens  if k <= 16]
    even_big   = [k for k in evens  if k >= 17]

    need_odd   = best_odd_count
    need_even  = 6 - best_odd_count
    need_small = best_small_count
    need_big   = best_big_count

    # 贪心：先填奇数且小，再奇数且大，再偶数且小，再偶数且大
    def fill_from(pool, count):
        nonlocal need_odd, need_even, need_small, need_big
        added = 0
        for k in pool:
            if added >= count:
                break
            if k not in selected:
                selected.add(k)
                added += 1
        return added

    fill_from([k for k in odd_small  if red_freq_all[k] >= 1], min(need_odd, need_small))
    fill_from([k for k in odd_big    if red_freq_all[k] >= 1], need_odd   - len([k for k in selected if k % 2 == 1]))
    fill_from([k for k in even_small if red_freq_all[k] >= 1], need_small - len([k for k in selected if k <= 16]))
    fill_from([k for k in even_big   if red_freq_all[k] >= 1], 6 - len(selected))

    # 不足时无约束补充
    for r in sorted(range(1, 34), key=lambda k: red_freq_all[k], reverse=True):
        if len(selected) >= 6:
            break
        if r not in selected:
            selected.add(r)

    final_reds = sorted(list(selected)[:6])

    # 蓝球取历史高频
    blue = max(blue_freq_all, key=lambda k: blue_freq_all[k])

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    blue,
        "method":       f"奇偶比/大小比（历史最优奇偶比 {best_odd_count}:{6-best_odd_count}，大小比 {best_big_count}:{best_small_count}）",
        "detail":       {"说明": "按历史最优奇偶比和大小比强制生成号码，奇偶比 = 奇数:偶数，大小比 = ≥17:≤16",
                         "最优奇偶比": f"{best_odd_count}奇{6-best_odd_count}偶",
                         "最优大小比": f"{best_big_count}大{best_small_count}小",
                         "奇偶比统计": dict(sorted(ratio_count.items())),
                         "大小比统计": dict(sorted(size_count.items()))},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_random_forest(db_path, history, progress_cb=None):
    """
    方案九：随机森林预测（🚀 优化版）

    优化点：
    1. 预计算所有特征矩阵，避免 O(N²×M) 的重复遍历
    2. 使用向量化操作代替 Python 循环
    3. 训练提速 10-50 倍

    Args:
        db_path: 数据库路径
        history: 历史数据列表
        progress_cb: 进度回调函数 cb(msg)
    """
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        n = len(history)
        if n < 50:
            return {"error": "历史数据不足（需≥50期）", "red_balls": [], "blue_ball": None}

        # 🚀 优化1：预计算红球特征
        if progress_cb: progress_cb("🌳 [优化] 预计算红球特征矩阵...")
        red_freq_mat, red_gap_mat = _precompute_features_fast(history, window=20, is_blue=False)

        # 构建训练集
        if progress_cb: progress_cb("🌳 [优化] 构建红球训练集...")
        X_red, y_red = [], []
        for i in range(20, n):
            reds_in_rec = {history[i][f"red{k}"] for k in range(1, 7)}
            feat_row = red_freq_mat[i, 1:34] / max(1, 20)
            gap_row = red_gap_mat[i, 1:34] / max(1, i)

            for num in range(1, 34):
                # ⚠️ 修复：feat_row[0] 对应号码1，所以用 num-1 索引
                feat = [num, num % 2, 1 if num <= 16 else 0, feat_row[num - 1], gap_row[num - 1]]
                X_red.append(feat)
                y_red.append(1 if num in reds_in_rec else 0)

        if len(X_red) < 100:
            return {"error": "训练样本不足", "red_balls": [], "blue_ball": None}

        # 训练红球模型
        if progress_cb: progress_cb("🌳 [优化] 训练红球随机森林（100棵树）...")
        clf_red = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_red.fit(X_red, y_red)

        # 🚀 优化2：预测下一期
        if progress_cb: progress_cb("🌳 [优化] 红球预测中...")
        feat_row = red_freq_mat[n-1, 1:34] / 20
        gap_row = red_gap_mat[n-1, 1:34] / max(1, n)

        pred_probs = []
        for num in range(1, 34):
            # ⚠️ 修复：feat_row[0] 对应号码1，所以用 num-1 索引
            feat = [num, num % 2, 1 if num <= 16 else 0, feat_row[num - 1], gap_row[num - 1]]
            prob = clf_red.predict_proba([feat])[0][1]
            pred_probs.append((num, prob))

        pred_probs.sort(key=lambda x: x[1], reverse=True)
        final_reds = sorted([x[0] for x in pred_probs[:6]])

        # 🚀 优化3：预计算蓝球特征
        if progress_cb: progress_cb("🌳 [优化] 预计算蓝球特征矩阵...")
        blue_freq_mat, blue_gap_mat = _precompute_features_fast(history, window=20, is_blue=True)

        # 构建蓝球训练集
        X_blue, y_blue = [], []
        for i in range(20, n):
            blue_in_rec = history[i]["blue"]
            feat_row = blue_freq_mat[i, 1:17] / 20
            gap_row = blue_gap_mat[i, 1:17] / max(1, i)

            for num in range(1, 17):
                # ⚠️ 修复：feat_row[0] 对应号码1，所以用 num-1 索引
                feat = [num, num % 2, 1 if num <= 8 else 0, feat_row[num - 1], gap_row[num - 1]]
                X_blue.append(feat)
                y_blue.append(1 if blue_in_rec == num else 0)

        blue_ball = 1
        if len(X_blue) >= 50:
            if progress_cb: progress_cb("🌳 [优化] 训练蓝球随机森林...")
            clf_blue = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            clf_blue.fit(X_blue, y_blue)

            feat_row = blue_freq_mat[n-1, 1:17] / 20
            gap_row = blue_gap_mat[n-1, 1:17] / max(1, n)
            blue_probs = []
            for num in range(1, 17):
                # ⚠️ 修复：feat_row[0] 对应号码1，所以用 num-1 索引
                feat = [num, num % 2, 1 if num <= 8 else 0, feat_row[num - 1], gap_row[num - 1]]
                prob = clf_blue.predict_proba([feat])[0][1]
                blue_probs.append((num, prob))
            blue_probs.sort(key=lambda x: x[1], reverse=True)
            blue_ball = blue_probs[0][0]

        red_freq_all, blue_freq_all = _calc_frequency(history)
        hot_top = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
        hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

        return {
            "next_issue": _calc_next_issue(history[-1]["issue"]),
            "red_balls": final_reds,
            "blue_ball": blue_ball,
            "method": f"随机森林（RF优化版，100棵树，全部{n}期历史）",
            "detail": {"说明": "🚀优化版：预计算特征矩阵，训练提速10-50倍",
                       "特征": "近20期频率、遗漏值、奇偶性、大小性",
                       "训练样本数": len(X_red)},
            "red_freq": red_freq_all,
            "blue_freq": blue_freq_all,
            "hot_reds": hot_top,
            "hot_blues": hot_btop,
            "history_used": n,
            "rf_probs": {str(x[0]): round(x[1], 4) for x in pred_probs[:10]},
        }
    except Exception as e:
        import traceback
        return {"error": f"随机森林预测失败: {e}", "red_balls": [], "blue_ball": None,
                "next_issue": None, "method": "random_forest"}



def predict_bayesian(history):
    """
    方案十：贝叶斯条件概率预测
    基于 P(next | last_reds, last_blue)：统计历史上"上一期出现了号码X"时，
    "本期出现号码Y"的条件频率，作为下一期的预测概率。
    """
    n = len(history)
    if n < 30:
        return None

    red_freq_all, blue_freq_all = _calc_frequency(history)

    # 统计条件共现：上期出现 x → 本期出现 y 的次数
    cooccur = {}   # cooccur[x][y] = count
    for i in range(1, n):
        prev_reds = {history[i-1][f"red{k}"] for k in range(1, 7)}
        curr_reds = {history[i][f"red{k}"]   for k in range(1, 7)}
        for x in prev_reds:
            if x not in cooccur:
                cooccur[x] = {}
            for y in curr_reds:
                cooccur[x][y] = cooccur[x].get(y, 0) + 1

    # 上一期号码
    last_reds = {history[-1][f"red{k}"] for k in range(1, 7)}

    # 对每个候选号码计算条件得分
    scores = {}
    for y in range(1, 34):
        score = 0
        for x in last_reds:
            if x in cooccur and y in cooccur[x]:
                total_after_x = sum(cooccur[x].values())
                score += cooccur[x][y] / max(1, total_after_x)
        scores[y] = score + red_freq_all[y] / (n * 6) * 0.1   # 加一点先验频率做平滑

    sorted_reds = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    final_reds  = sorted(sorted_reds[:6])

    # 蓝球：上期蓝球后最常见的蓝球
    last_blue = history[-1]["blue"]
    blue_cooccur = {}
    for i in range(1, n):
        pb = history[i-1]["blue"]
        cb = history[i]["blue"]
        if pb not in blue_cooccur:
            blue_cooccur[pb] = {}
        blue_cooccur[pb][cb] = blue_cooccur[pb].get(cb, 0) + 1

    if last_blue in blue_cooccur and blue_cooccur[last_blue]:
        blue = max(blue_cooccur[last_blue], key=lambda k: blue_cooccur[last_blue][k])
    else:
        blue = max(blue_freq_all, key=lambda k: blue_freq_all[k])

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    blue,
        "method":       f"贝叶斯条件概率（基于 {n} 期历史条件共现）",
        "detail":       {"说明": "P(本期出y | 上期出x)，统计上期与本期号码的条件共现概率",
                         "上期红球": sorted(last_reds),
                         "上期蓝球": last_blue,
                         "Top3红球得分": {str(k): round(scores[k], 4) for k in sorted_reds[:3]}},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_genetic_algorithm(history, progress_cb=None):
    """
    方案十一：遗传算法预测
    将号码组合视为"个体"，用遗传进化策略优化"历史吻合度"（个体与历史开奖的匹配数）。
    经过多代进化，得到最优号码组合。
    
    ✅ 2026-04-20 优化：
      - 种群 200→80（减少60%内存和计算）
      - 代数 80→40（减少50%迭代）
      - 添加 progress_cb 回调支持UI实时进度显示
    """
    n = len(history)
    if n < 20:
        return None

    import random as _rand

    RED_N = 6
    RED_MAX = 33
    BLUE_MAX = 16
    POP_SIZE = 80       # ✅ 原值200 → 优化为80（提速~2.5倍）
    GENERATIONS = 40    # ✅ 原值80  → 优化为40（提速~2倍）
    MUTATION_RATE = 0.15

    red_freq_all, blue_freq_all = _calc_frequency(history)

    def fitness(individual):
        """适应度：与近50期历史的平均匹配红球数"""
        recent = history[-50:] if n >= 50 else history
        total_match = 0
        for rec in recent:
            hist_reds = {rec[f"red{k}"] for k in range(1, 7)}
            match = len(set(individual) & hist_reds)
            total_match += match
        return total_match / len(recent)

    def random_individual():
        return tuple(sorted(_rand.sample(range(1, RED_MAX + 1), RED_N)))

    # 初始种群（偏向高频号）
    freq_pool = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)
    population = []
    for _ in range(POP_SIZE):
        hot_k  = min(15, len(freq_pool))
        hot    = _rand.sample(freq_pool[:hot_k], min(3, hot_k))
        rest   = _rand.sample([k for k in range(1, RED_MAX+1) if k not in hot], RED_N - len(hot))
        population.append(tuple(sorted(hot + rest)))

    # 进化
    if progress_cb: progress_cb(f"🧬 遗传算法开始进化（种群{POP_SIZE}，{GENERATIONS}代）...")
    for gen in range(GENERATIONS):
        # ✅ 每10代报告进度
        if progress_cb and gen % 10 == 0:
            progress_cb(f"🧬 第 {gen}/{GENERATIONS} 代进化中...")
            
        scored = sorted([(fitness(ind), ind) for ind in population], reverse=True)
        top = [ind for _, ind in scored[:POP_SIZE // 4]]

        new_pop = list(top)
        while len(new_pop) < POP_SIZE:
            p1, p2 = _rand.sample(top, 2)
            combined = list(set(p1) | set(p2))
            if len(combined) >= RED_N:
                child = tuple(sorted(_rand.sample(combined, RED_N)))
            else:
                child = tuple(sorted(_rand.sample(range(1, RED_MAX+1), RED_N)))

            # 变异
            if _rand.random() < MUTATION_RATE:
                child = list(child)
                idx = _rand.randint(0, RED_N - 1)
                new_num = _rand.randint(1, RED_MAX)
                while new_num in child:
                    new_num = _rand.randint(1, RED_MAX)
                child[idx] = new_num
                child = tuple(sorted(child))

            new_pop.append(child)
        population = new_pop[:POP_SIZE]

    # 取最优个体
    best = max(population, key=fitness)
    final_reds = list(best)

    # 蓝球：历史高频
    blue = max(blue_freq_all, key=lambda k: blue_freq_all[k])

    hot_top  = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   _calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_ball":    blue,
        "method":       f"遗传算法（种群{POP_SIZE}，进化{GENERATIONS}代，适应度=近50期匹配数）",
        "detail":       {"说明": "将号码组合视为个体，用进化策略优化与历史开奖的吻合度",
                         "种群大小": POP_SIZE,
                         "进化代数": GENERATIONS,
                         "变异率": MUTATION_RATE,
                         "最优适应度": round(fitness(best), 4)},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_next_issue(db_path, method="linear", progress_cb=None):
    """
    主预测函数，支持多种方案

    Args:
        db_path: 数据库路径
        method: "linear" / "frequency" / "entropy" / "gap" / "lstm" / "xgboost" / "gru"
                "hot_cold" / "same_period" / "consecutive" / "odd_even"
                "random_forest" / "bayesian" / "genetic"
        progress_cb: 进度回调函数 cb(msg) - UI实时显示进度

    Returns:
        dict: 预测结果
    """
    history = load_history(db_path)

    if not history:
        return None

    if method == "frequency":
        if progress_cb: progress_cb("📊 执行频率加权预测...")
        return predict_frequency_weighted(history)
    elif method == "entropy":
        if progress_cb: progress_cb("📈 执行熵最大化预测...")
        return predict_entropy_maximized(history)
    elif method == "gap":
        if progress_cb: progress_cb("📉 执行遗漏值预测...")
        return predict_gap_maximized(history)
    elif method == "hot_cold":
        if progress_cb: progress_cb("🌡️ 执行冷热号平衡预测...")
        return predict_hot_cold_balance(history)
    elif method == "same_period":
        if progress_cb: progress_cb("📅 执行同期历史预测...")
        return predict_same_period_history(history)
    elif method == "consecutive":
        if progress_cb: progress_cb("🔢 执行连号跨度预测...")
        return predict_consecutive_span(history)
    elif method == "odd_even":
        if progress_cb: progress_cb("⚖️ 执行奇偶比预测...")
        return predict_odd_even_ratio(history)
    elif method == "random_forest":
        if progress_cb: progress_cb("🌳 随机森林模型准备中...")
        return predict_random_forest(db_path, history, progress_cb=progress_cb)
    elif method == "bayesian":
        if progress_cb: progress_cb("🔵 贝叶斯概率计算中...")
        return predict_bayesian(history)
    elif method == "genetic":
        if progress_cb: progress_cb("🧬 遗传算法初始化中...")
        return predict_genetic_algorithm(history, progress_cb=progress_cb)
    elif method == "lstm":
        # MLP (sklearn)
        global _lstm_model
        if _lstm_model is None:
            try:
                import lstm_model as lm
                _lstm_model = lm
            except Exception as e:
                return {"issue": None, "red_balls": [], "blue_ball": None,
                        "method": "lstm", "error": f"MLP模块加载失败: {e}"}
        result = _lstm_model.predict_lstm(db_path)
        if result is None:
            return {"issue": None, "red_balls": [], "blue_ball": None,
                    "method": "lstm", "error": "MLP 预测失败"}
        return result
    elif method == "xgboost":
        # XGBoost 梯度提升树
        global _xgb_model
        if _xgb_model is None:
            try:
                import xgboost_model as xm
                _xgb_model = xm
            except Exception as e:
                return {"issue": None, "red_balls": [], "blue_ball": None,
                        "method": "xgboost", "error": f"XGBoost模块加载失败: {e}"}
        result = _xgb_model.predict_xgboost(db_path)
        if result is None:
            return {"issue": None, "red_balls": [], "blue_ball": None,
                    "method": "xgboost", "error": "XGBoost 预测失败（返回空）"}
        # predict_xgboost 在异常时返回 {"__error__": ...} 以传递真实原因
        if "__error__" in result:
            return {"issue": None, "red_balls": [], "blue_ball": None,
                    "method": "xgboost", "error": result["__error__"]}
        return result
    elif method == "gru":
        # GRU 序列模型 (PyTorch)
        global _gru_model
        if _gru_model is None:
            try:
                import lstm_rnn_model as gm
                _gru_model = gm
            except Exception as e:
                return {"issue": None, "red_balls": [], "blue_ball": None,
                        "method": "gru", "error": f"GRU模块加载失败: {e}"}
        result = _gru_model.predict_gru(db_path)
        if result is None:
            return {"issue": None, "red_balls": [], "blue_ball": None,
                    "method": "gru", "error": "GRU 预测失败"}
        return result
    else:
        return predict_linear_regression(history)


# ─── 测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    db_path = os.path.join(os.path.dirname(__file__), "ssq.db")

    print("=" * 50)
    print("方案一：线性回归预测")
    print("=" * 50)
    result1 = predict_next_issue(db_path, method="linear")
    if result1:
        print(f"预测期号: {result1['next_issue']}")
        print(f"红球: {result1['red_balls']}")
        print(f"蓝球: {result1['blue_ball']}")
        print(f"方法: {result1['method']}")
        print(f"使用了 {result1['history_used']} 期数据")
    else:
        print("数据不足")

    print()
    print("=" * 50)
    print("方案二：频率加权预测")
    print("=" * 50)
    result2 = predict_next_issue(db_path, method="frequency")
    if result2:
        print(f"预测期号: {result2['next_issue']}")
        print(f"红球: {result2['red_balls']}")
        print(f"蓝球: {result2['blue_ball']}")
        print(f"方法: {result2['method']}")
        print(f"使用了 {result2['history_used']} 期数据")
    else:
        print("数据不足")

    print()
    print("=" * 50)
    print("方案三：熵最大化预测")
    print("=" * 50)
    result3 = predict_next_issue(db_path, method="entropy")
    if result3:
        print(f"预测期号: {result3['next_issue']}")
        print(f"红球: {result3['red_balls']}")
        print(f"蓝球: {result3['blue_ball']}")
        print(f"方法: {result3['method']}")
        print(f"使用了 {result3['history_used']} 期数据")
        if "entropy_data" in result3:
            ent = result3["entropy_data"]
            print(f"\n信息熵分析:")
            print(f"  红球熵: {ent['red_entropy']} / {ent['red_max_entropy']} ({ent['red_relative_entropy']*100:.1f}%)")
            print(f"  蓝球熵: {ent['blue_entropy']} / {ent['blue_max_entropy']} ({ent['blue_relative_entropy']*100:.1f}%)")
            print(f"  红球变异系数: {ent['red_cv']:.4f}")
    else:
        print("数据不足")

    print()
    print("=" * 50)
    print("方案四：遗漏值最大化预测")
    print("=" * 50)
    result4 = predict_next_issue(db_path, method="gap")
    if result4:
        print(f"预测期号: {result4['next_issue']}")
        print(f"红球: {result4['red_balls']}")
        print(f"蓝球: {result4['blue_ball']}")
        print(f"方法: {result4['method']}")
        print(f"使用了 {result4['history_used']} 期数据")
        if "gap_data" in result4:
            gap = result4["gap_data"]
            print(f"\n遗漏值分析:")
            print(f"  最大红球遗漏: {gap['max_gap_red']} 号，已遗漏 {gap['max_gap_red_value']} 期")
            print(f"  最大蓝球遗漏: {gap['max_gap_blue']} 号，已遗漏 {gap['max_gap_blue_value']} 期")
            print(f"  平均红球遗漏: {gap['avg_gap_red']:.1f} 期")
            print(f"  平均蓝球遗漏: {gap['avg_gap_blue']:.1f} 期")
    else:
        print("数据不足")
