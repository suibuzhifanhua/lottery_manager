"""
大乐透预测模块
规则：从1-35中选5个前区号（红球）+ 从1-12中选2个后区号（蓝球）
提供多种预测方案：
1. 线性回归预测
2. 频率加权预测
3. 熵最大化预测
4. 遗漏值最大化预测
5. MLP神经网络
6. XGBoost
7. GRU序列模型
"""
import random
import sqlite3

# 大乐透规则
DLT_RED_COUNT  = 35   # 前区：1-35，选5个
DLT_BLUE_COUNT = 12   # 后区：1-12，选2个
DLT_RED_PICK   = 5
DLT_BLUE_PICK  = 2


# ─── 数据库读取 ────────────────────────────────────────────────
def load_history(db_path):
    """
    读取全部大乐透历史记录（按期号升序）
    返回 list of dict: {issue, red1..red5, blue1, blue2}
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT issue, red1, red2, red3, red4, red5, blue1, blue2
            FROM dlt_history
            ORDER BY issue ASC
        """)
        rows = c.fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        result.append({
            "issue": row[0],
            "red1": row[1], "red2": row[2], "red3": row[3],
            "red4": row[4], "red5": row[5],
            "blue1": row[6], "blue2": row[7],
        })
    return result


def load_history_tuples(db_path):
    """返回 list of tuple: (issue, red1..red5, blue1, blue2)"""
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT issue, red1, red2, red3, red4, red5, blue1, blue2
            FROM dlt_history ORDER BY issue ASC
        """)
        rows = c.fetchall()
    finally:
        conn.close()
    return rows


# ─── 期号推算 ──────────────────────────────────────────────────
def calc_next_issue(last_issue):
    """根据最后一期推算下一期号"""
    try:
        s = str(last_issue)
        if len(s) == 7:
            # 格式：2025001
            year = int(s[:4])
            num  = int(s[4:])
        else:
            # 格式：25001
            year = int(s[:2])
            num  = int(s[2:])
        if num >= 180:
            year += 1
            num = 1
        else:
            num += 1
        if len(s) == 7:
            return f"{year:04d}{num:03d}"
        else:
            return f"{year:02d}{num:03d}"
    except Exception:
        return str(int(last_issue) + 1)


# ─── 频率统计 ──────────────────────────────────────────────────
def _calc_frequency(history):
    """计算红球/蓝球历史频率"""
    red_freq  = {i: 0 for i in range(1, DLT_RED_COUNT  + 1)}
    blue_freq = {i: 0 for i in range(1, DLT_BLUE_COUNT + 1)}
    for rec in history:
        for i in range(1, DLT_RED_PICK + 1):
            red_freq[rec[f"red{i}"]] += 1
        for i in range(1, DLT_BLUE_PICK + 1):
            blue_freq[rec[f"blue{i}"]] += 1
    return red_freq, blue_freq


# ─── 遗漏值计算 ────────────────────────────────────────────────
def _compute_gap(history):
    """
    计算所有号码当前遗漏值（O(n) 单次反向扫描）
    history: list of dict
    返回 (red_gap, blue_gap)
    """
    n = len(history)
    red_gap  = {i: n for i in range(1, DLT_RED_COUNT  + 1)}
    blue_gap = {i: n for i in range(1, DLT_BLUE_COUNT + 1)}
    red_found  = set()
    blue_found = set()

    for offset, rec in enumerate(reversed(history)):
        gap = offset
        reds  = {rec[f"red{i}"]  for i in range(1, DLT_RED_PICK  + 1)}
        blues = {rec[f"blue{i}"] for i in range(1, DLT_BLUE_PICK + 1)}

        for r in reds:
            if r not in red_found:
                red_gap[r] = gap
                red_found.add(r)
        for b in blues:
            if b not in blue_found:
                blue_gap[b] = gap
                blue_found.add(b)

        if len(red_found) == DLT_RED_COUNT and len(blue_found) == DLT_BLUE_COUNT:
            break

    return red_gap, blue_gap


# ─── 线性回归 ──────────────────────────────────────────────────
def _linear_regression(x_list, y_list):
    n = len(x_list)
    if n == 0:
        return 0, 0
    sx  = sum(x_list)
    sy  = sum(y_list)
    sxy = sum(x * y for x, y in zip(x_list, y_list))
    sx2 = sum(x * x for x in x_list)
    denom = n * sx2 - sx * sx
    if denom == 0:
        return 0, sy / n
    k = (n * sxy - sx * sy) / denom
    b = (sy - k * sx) / n
    return k, b


def predict_linear_regression(history):
    """方案一：线性回归预测"""
    n = len(history)
    if n < 10:
        return None

    x = list(range(n))
    next_x = n

    # 5个红球位置回归
    raw_reds = []
    detail = {}
    for i in range(1, DLT_RED_PICK + 1):
        y = [rec[f"red{i}"] for rec in history]
        k, b = _linear_regression(x, y)
        pred = k * next_x + b
        raw_reds.append(round(pred))
        detail[f"前区位置{i}原始预测"] = round(pred, 2)

    # 2个蓝球位置回归
    raw_blues = []
    for i in range(1, DLT_BLUE_PICK + 1):
        y = [rec[f"blue{i}"] for rec in history]
        k, b = _linear_regression(x, y)
        pred = k * next_x + b
        raw_blues.append(round(pred))
        detail[f"后区位置{i}原始预测"] = round(pred, 2)

    # 截断到合法范围
    raw_reds  = [max(1, min(DLT_RED_COUNT,  r)) for r in raw_reds]
    raw_blues = [max(1, min(DLT_BLUE_COUNT, b)) for b in raw_blues]

    red_freq, blue_freq = _calc_frequency(history)

    # 红球去重并补充到5个
    seen = set()
    final_reds = []
    for r in raw_reds:
        if r not in seen and 1 <= r <= DLT_RED_COUNT:
            final_reds.append(r)
            seen.add(r)
    if len(final_reds) < DLT_RED_PICK:
        sorted_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)
        for r in sorted_reds:
            if r not in seen and len(final_reds) < DLT_RED_PICK:
                final_reds.append(r)
                seen.add(r)
    final_reds.sort()

    # 蓝球去重并补充到2个
    seen_blue = set()
    final_blues = []
    for b in raw_blues:
        if b not in seen_blue and 1 <= b <= DLT_BLUE_COUNT:
            final_blues.append(b)
            seen_blue.add(b)
    if len(final_blues) < DLT_BLUE_PICK:
        sorted_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)
        for b in sorted_blues:
            if b not in seen_blue and len(final_blues) < DLT_BLUE_PICK:
                final_blues.append(b)
                seen_blue.add(b)
    final_blues.sort()

    hot_reds  = sorted(red_freq.keys(),  key=lambda k: red_freq[k],  reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"线性回归（全部 {n} 期历史数据）",
        "detail":       detail,
        "red_freq":     red_freq,
        "blue_freq":    blue_freq,
        "hot_reds":     hot_reds,
        "hot_blues":    hot_blues,
        "history_used": n,
    }


def predict_frequency_weighted(history):
    """方案二：频率加权预测"""
    n = len(history)
    if n < 10:
        return None

    red_freq, blue_freq = _calc_frequency(history)

    def weighted_sample(pool, freq, count):
        weights = [freq[i] for i in pool]
        if sum(weights) == 0:
            weights = [1] * len(pool)
        selected = []
        tmp_pool = pool.copy()
        tmp_w    = weights.copy()
        for _ in range(count):
            total = sum(tmp_w)
            if total == 0:
                choice = random.choice(tmp_pool)
            else:
                r = random.uniform(0, total)
                cumsum = 0
                choice = tmp_pool[-1]
                for num, w in zip(tmp_pool, tmp_w):
                    cumsum += w
                    if r <= cumsum:
                        choice = num
                        break
            selected.append(choice)
            idx = tmp_pool.index(choice)
            tmp_pool.pop(idx)
            tmp_w.pop(idx)
        return selected

    selected_reds  = sorted(weighted_sample(list(range(1, DLT_RED_COUNT  + 1)), red_freq,  DLT_RED_PICK))
    selected_blues = sorted(weighted_sample(list(range(1, DLT_BLUE_COUNT + 1)), blue_freq, DLT_BLUE_PICK))

    hot_reds  = sorted(red_freq.keys(),  key=lambda k: red_freq[k],  reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    selected_reds,
        "blue_balls":   selected_blues,
        "method":       f"频率加权随机（全部 {n} 期历史数据）",
        "detail":       {"说明": "基于历史出现频率加权随机选择，不放回抽样"},
        "red_freq":     red_freq,
        "blue_freq":    blue_freq,
        "hot_reds":     hot_reds,
        "hot_blues":    hot_blues,
        "history_used": n,
    }


def predict_entropy_maximized(history):
    """方案三：熵最大化预测（选最冷门号码）"""
    n = len(history)
    if n < 10:
        return None

    red_freq, blue_freq = _calc_frequency(history)

    expected_red  = n * DLT_RED_PICK  / DLT_RED_COUNT
    expected_blue = n * DLT_BLUE_PICK / DLT_BLUE_COUNT

    red_dev  = {k: expected_red  - v for k, v in red_freq.items()}
    blue_dev = {k: expected_blue - v for k, v in blue_freq.items()}

    selected_reds  = sorted(sorted(red_dev.keys(),  key=lambda k: red_dev[k],  reverse=True)[:DLT_RED_PICK])
    selected_blues = sorted(sorted(blue_dev.keys(), key=lambda k: blue_dev[k], reverse=True)[:DLT_BLUE_PICK])

    hot_reds  = sorted(red_freq.keys(),  key=lambda k: red_freq[k],  reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    selected_reds,
        "blue_balls":   selected_blues,
        "method":       f"熵最大化（全部 {n} 期历史数据）",
        "detail":       {"说明": "选择历史出现频率最低的号码，使分布趋向均匀",
                         "红球策略": f"选最冷门的 {DLT_RED_PICK} 个前区号",
                         "蓝球策略": f"选最冷门的 {DLT_BLUE_PICK} 个后区号"},
        "red_freq":     red_freq,
        "blue_freq":    blue_freq,
        "hot_reds":     hot_reds,
        "hot_blues":    hot_blues,
        "history_used": n,
    }


def predict_gap_maximized(history):
    """方案四：遗漏值最大化预测"""
    n = len(history)
    if n < 10:
        return None

    red_gap, blue_gap = _compute_gap(history)

    sorted_reds  = sorted(red_gap.keys(),  key=lambda k: red_gap[k],  reverse=True)
    sorted_blues = sorted(blue_gap.keys(), key=lambda k: blue_gap[k], reverse=True)

    selected_reds  = sorted(sorted_reds[:DLT_RED_PICK])
    selected_blues = sorted(sorted_blues[:DLT_BLUE_PICK])

    red_freq, blue_freq = _calc_frequency(history)
    hot_reds  = sorted(red_freq.keys(),  key=lambda k: red_freq[k],  reverse=True)[:10]
    hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]

    max_red_num  = sorted_reds[0]
    max_blue_num = sorted_blues[0]

    gap_data = {
        "red_gap":            red_gap,
        "blue_gap":           blue_gap,
        "max_gap_red":        max_red_num,
        "max_gap_blue":       max_blue_num,
        "max_gap_red_value":  red_gap[max_red_num],
        "max_gap_blue_value": blue_gap[max_blue_num],
        "avg_gap_red":        sum(red_gap.values())  / len(red_gap),
        "avg_gap_blue":       sum(blue_gap.values()) / len(blue_gap),
    }

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    selected_reds,
        "blue_balls":   selected_blues,
        "method":       f"遗漏值最大化（全部 {n} 期历史数据）",
        "detail":       {"说明": "选当前遗漏值最大（最久未出）的号码",
                         "最大前区遗漏": gap_data["max_gap_red_value"],
                         "最大后区遗漏": gap_data["max_gap_blue_value"]},
        "red_freq":     red_freq,
        "blue_freq":    blue_freq,
        "hot_reds":     hot_reds,
        "hot_blues":    hot_blues,
        "history_used": n,
        "gap_data":     gap_data,
        "max_gap_reds": sorted_reds[:10],
    }


def predict_hot_cold_balance(history):
    """
    方案五：冷热号平衡预测（大乐透）
    近20期：热号(频繁) + 冷号(未出)，按 3热2冷 + 后区1热1冷 比例混合。
    """
    n = len(history)
    if n < 20:
        return None

    recent = history[-20:]
    red_freq_all, blue_freq_all = _calc_frequency(history)
    red_freq_rec, blue_freq_rec = _calc_frequency(recent)

    hot_reds  = [k for k, v in red_freq_rec.items() if v >= 2]
    cold_reds = [k for k, v in red_freq_rec.items() if v == 0]
    hot_blues = [k for k, v in blue_freq_rec.items() if v >= 1]
    cold_blues = [k for k, v in blue_freq_rec.items() if v == 0]

    if len(hot_reds) < 3 or len(cold_reds) < 2:
        return predict_frequency_weighted(history)

    selected = set(random.sample(hot_reds, min(3, len(hot_reds))))
    cold_pool = [r for r in cold_reds if r not in selected]
    while len(selected) < DLT_RED_PICK and cold_pool:
        c = random.choice(cold_pool)
        selected.add(c)
        cold_pool.remove(c)
    while len(selected) < DLT_RED_PICK:
        r = random.choice([x for x in range(1, DLT_RED_COUNT+1) if x not in selected])
        selected.add(r)
    final_reds = sorted(selected)

    # 后区：1热1冷
    final_blues = []
    if hot_blues:
        final_blues.append(random.choice(hot_blues))
    if cold_blues:
        c = random.choice([b for b in cold_blues if b not in final_blues])
        final_blues.append(c)
    while len(final_blues) < DLT_BLUE_PICK:
        b = random.choice([x for x in range(1, DLT_BLUE_COUNT+1) if x not in final_blues])
        final_blues.append(b)
    final_blues = sorted(final_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       "冷热号平衡（近20期热号+冷号混合）",
        "detail":       {"说明": "近20期频繁出现为热号，未出现为冷号，按3热2冷比例混合",
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
    方案六：同期历史预测（大乐透）
    统计历史上同月份开奖的号码，找出高频号码。
    """
    import datetime
    n = len(history)
    if n < 30:
        return None

    try:
        now_month = datetime.date.today().month
    except Exception:
        now_month = 1

    def issue_to_approx_month(issue_str):
        try:
            s = str(issue_str)
            seq = int(s[4:]) if len(s) == 7 else int(s[2:])
            return min(12, max(1, (seq - 1) // 10 + 1))
        except Exception:
            return 0

    same_month_recs = [r for r in history if issue_to_approx_month(r["issue"]) == now_month]
    if len(same_month_recs) < 5:
        same_month_recs = history[-30:]

    red_freq_sm, blue_freq_sm = _calc_frequency(same_month_recs)
    red_freq_all, blue_freq_all = _calc_frequency(history)

    sorted_reds  = sorted(red_freq_sm.keys(),  key=lambda k: red_freq_sm[k],  reverse=True)
    sorted_blues = sorted(blue_freq_sm.keys(), key=lambda k: blue_freq_sm[k], reverse=True)

    final_reds  = sorted(sorted_reds[:DLT_RED_PICK])
    final_blues = sorted(sorted_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"同期历史（{now_month}月高频号码，{len(same_month_recs)}期参考）",
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
    方案七：连号跨度分析（大乐透）
    以历史最常见连号对为锚点，按平均跨度补充剩余号码。
    """
    n = len(history)
    if n < 20:
        return None

    red_freq_all, blue_freq_all = _calc_frequency(history)

    spans = [history[i][f"red{DLT_RED_PICK}"] - history[i]["red1"] for i in range(n)]
    avg_span = sum(spans) / len(spans)
    target_span = round(avg_span)

    consec_freq = {i: 0 for i in range(1, DLT_RED_COUNT)}
    for rec in history:
        reds = {rec[f"red{j}"] for j in range(1, DLT_RED_PICK+1)}
        for r in reds:
            if r + 1 in reds:
                consec_freq[r] += 1

    best_consec = max(consec_freq, key=lambda k: consec_freq[k])
    selected = {best_consec, best_consec + 1}
    weights = {k: red_freq_all[k] for k in range(1, DLT_RED_COUNT+1) if k not in selected}
    pool = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)

    for r in pool:
        if len(selected) >= DLT_RED_PICK:
            break
        lo = max(1, best_consec - target_span // 2)
        hi = min(DLT_RED_COUNT, best_consec + target_span)
        if lo <= r <= hi:
            selected.add(r)
    for r in pool:
        if len(selected) >= DLT_RED_PICK:
            break
        if r not in selected:
            selected.add(r)

    final_reds = sorted(list(selected)[:DLT_RED_PICK])

    sorted_blues = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)
    final_blues = sorted(sorted_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"连号跨度分析（历史均跨{round(avg_span,1)}，连号锚{best_consec}-{best_consec+1}）",
        "detail":       {"说明": "以历史最常见连号对为锚点，按平均跨度范围补充剩余号码",
                         "连号锚点": f"{best_consec}-{best_consec+1}",
                         "历史平均跨度": round(avg_span, 1)},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_odd_even_ratio(history):
    """
    方案八：奇偶比/大小比预测（大乐透）
    大乐透前区历史最优奇偶比，强制生成满足比例的号码。
    """
    n = len(history)
    if n < 10:
        return None

    ratio_count = {}
    for rec in history:
        reds = [rec[f"red{i}"] for i in range(1, DLT_RED_PICK+1)]
        odd_c = sum(1 for r in reds if r % 2 == 1)
        ratio_count[odd_c] = ratio_count.get(odd_c, 0) + 1
    best_odd = max(ratio_count, key=lambda k: ratio_count[k])

    size_count = {}
    for rec in history:
        reds = [rec[f"red{i}"] for i in range(1, DLT_RED_PICK+1)]
        small_c = sum(1 for r in reds if r <= 17)
        size_count[small_c] = size_count.get(small_c, 0) + 1
    best_small = max(size_count, key=lambda k: size_count[k])
    best_big = DLT_RED_PICK - best_small

    red_freq_all, blue_freq_all = _calc_frequency(history)

    odds   = sorted([k for k in range(1, DLT_RED_COUNT+1) if k % 2 == 1], key=lambda k: red_freq_all[k], reverse=True)
    evens  = sorted([k for k in range(1, DLT_RED_COUNT+1) if k % 2 == 0], key=lambda k: red_freq_all[k], reverse=True)
    smalls = sorted([k for k in range(1, DLT_RED_COUNT+1) if k <= 17],    key=lambda k: red_freq_all[k], reverse=True)
    bigs   = sorted([k for k in range(1, DLT_RED_COUNT+1) if k >= 18],    key=lambda k: red_freq_all[k], reverse=True)

    selected = set()
    odd_small  = [k for k in odds  if k <= 17]
    odd_big    = [k for k in odds  if k >= 18]
    even_small = [k for k in evens if k <= 17]
    even_big   = [k for k in evens if k >= 18]

    def fill_from(pool, count):
        added = 0
        for k in pool:
            if added >= count or len(selected) >= DLT_RED_PICK:
                break
            if k not in selected:
                selected.add(k)
                added += 1

    fill_from(odd_small,  min(best_odd, best_small))
    fill_from(odd_big,    best_odd - len([k for k in selected if k % 2 == 1]))
    fill_from(even_small, best_small - len([k for k in selected if k <= 17]))
    fill_from(even_big,   DLT_RED_PICK - len(selected))
    for r in sorted(range(1, DLT_RED_COUNT+1), key=lambda k: red_freq_all[k], reverse=True):
        if len(selected) >= DLT_RED_PICK:
            break
        if r not in selected:
            selected.add(r)

    final_reds = sorted(list(selected)[:DLT_RED_PICK])

    # 后区：高频后区号
    sorted_blues = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)
    final_blues = sorted(sorted_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"奇偶比/大小比（历史最优 {best_odd}奇{DLT_RED_PICK-best_odd}偶，{best_big}大{best_small}小）",
        "detail":       {"说明": "按历史最优奇偶比和大小比强制生成号码",
                         "最优奇偶比": f"{best_odd}奇{DLT_RED_PICK-best_odd}偶",
                         "最优大小比": f"{best_big}大{best_small}小",
                         "奇偶比统计": dict(sorted(ratio_count.items())),
                         "大小比统计": dict(sorted(size_count.items()))},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_random_forest(db_path, history):
    """
    方案九：随机森林预测（大乐透）
    使用 sklearn RandomForestClassifier 对每个前区号码做二分类。
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        import os, sys

        n = len(history)
        if n < 50:
            return {"error": "历史数据不足（需≥50期）", "red_balls": [], "blue_balls": []}

        def build_features(hist, idx, num, is_blue=False, blue_idx=0):
            past = hist[:idx]
            if not past:
                return None
            window = past[-20:] if len(past) >= 20 else past
            if not is_blue:
                freq = sum(1 for r in window
                           for k in range(1, DLT_RED_PICK+1)
                           if r[f"red{k}"] == num)
            else:
                freq = sum(1 for r in window if r[f"blue{blue_idx+1}"] == num)

            gap = 0
            for j in range(len(past) - 1, -1, -1):
                rec = past[j]
                if not is_blue:
                    nums_in = {rec[f"red{k}"] for k in range(1, DLT_RED_PICK+1)}
                else:
                    nums_in = {rec[f"blue{blue_idx+1}"]}
                if num in nums_in:
                    break
                gap += 1
            else:
                gap = len(past)
            return [num, num % 2, 1 if num <= (17 if not is_blue else 6) else 0,
                    freq / max(1, len(window)), gap / max(1, len(past))]

        X_red, y_red = [], []
        for i in range(20, n):
            reds_in = {history[i][f"red{k}"] for k in range(1, DLT_RED_PICK+1)}
            for num in range(1, DLT_RED_COUNT+1):
                feat = build_features(history, i, num, is_blue=False)
                if feat:
                    X_red.append(feat)
                    y_red.append(1 if num in reds_in else 0)

        if len(X_red) < 100:
            return {"error": "训练样本不足", "red_balls": [], "blue_balls": []}

        clf_red = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_red.fit(X_red, y_red)

        pred_probs = []
        for num in range(1, DLT_RED_COUNT+1):
            feat = build_features(history, n, num, is_blue=False)
            if feat:
                prob = clf_red.predict_proba([feat])[0][1]
                pred_probs.append((num, prob))
        pred_probs.sort(key=lambda x: x[1], reverse=True)
        final_reds = sorted([x[0] for x in pred_probs[:DLT_RED_PICK]])

        # 后区
        red_freq_all, blue_freq_all = _calc_frequency(history)
        sorted_blues = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)
        final_blues = sorted(sorted_blues[:DLT_BLUE_PICK])

        hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
        hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

        return {
            "next_issue":   calc_next_issue(history[-1]["issue"]),
            "red_balls":    final_reds,
            "blue_balls":   final_blues,
            "method":       f"随机森林（RF，100棵树，全部{n}期历史）",
            "detail":       {"说明": "RandomForest对每个前区号码做二分类，取概率最高的5个",
                             "特征": "近20期频率、遗漏值、奇偶性、大小性",
                             "训练样本数": len(X_red)},
            "red_freq":     red_freq_all,
            "blue_freq":    blue_freq_all,
            "hot_reds":     hot_top,
            "hot_blues":    hot_btop,
            "history_used": n,
            "rf_probs":     {str(x[0]): round(x[1], 4) for x in pred_probs[:10]},
        }
    except Exception as e:
        return {"error": f"随机森林预测失败: {e}", "red_balls": [], "blue_balls": [],
                "next_issue": None, "method": "random_forest"}


def predict_bayesian(history):
    """
    方案十：贝叶斯条件概率预测（大乐透）
    基于上一期前区号码与本期号码的条件共现频率进行预测。
    """
    n = len(history)
    if n < 30:
        return None

    red_freq_all, blue_freq_all = _calc_frequency(history)

    cooccur = {}
    for i in range(1, n):
        prev_reds = {history[i-1][f"red{k}"] for k in range(1, DLT_RED_PICK+1)}
        curr_reds = {history[i][f"red{k}"]   for k in range(1, DLT_RED_PICK+1)}
        for x in prev_reds:
            if x not in cooccur:
                cooccur[x] = {}
            for y in curr_reds:
                cooccur[x][y] = cooccur[x].get(y, 0) + 1

    last_reds = {history[-1][f"red{k}"] for k in range(1, DLT_RED_PICK+1)}
    scores = {}
    for y in range(1, DLT_RED_COUNT+1):
        score = 0
        for x in last_reds:
            if x in cooccur and y in cooccur[x]:
                total = sum(cooccur[x].values())
                score += cooccur[x][y] / max(1, total)
        scores[y] = score + red_freq_all[y] / (n * DLT_RED_PICK) * 0.1

    sorted_reds = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    final_reds  = sorted(sorted_reds[:DLT_RED_PICK])

    sorted_blues = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)
    final_blues  = sorted(sorted_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"贝叶斯条件概率（基于{n}期历史条件共现）",
        "detail":       {"说明": "P(本期出y | 上期出x)，统计上期与本期号码的条件共现概率",
                         "上期前区": sorted(last_reds),
                         "Top3红球得分": {str(k): round(scores[k], 4) for k in sorted_reds[:3]}},
        "red_freq":     red_freq_all,
        "blue_freq":    blue_freq_all,
        "hot_reds":     hot_top,
        "hot_blues":    hot_btop,
        "history_used": n,
    }


def predict_genetic_algorithm(history):
    """
    方案十一：遗传算法预测（大乐透）
    用遗传进化策略优化前区号码组合的历史吻合度。
    """
    n = len(history)
    if n < 20:
        return None

    import random as _rand

    POP_SIZE = 200
    GENERATIONS = 80
    MUTATION_RATE = 0.15

    red_freq_all, blue_freq_all = _calc_frequency(history)

    def fitness(individual):
        recent = history[-50:] if n >= 50 else history
        total_match = 0
        for rec in recent:
            hist_reds = {rec[f"red{k}"] for k in range(1, DLT_RED_PICK+1)}
            total_match += len(set(individual) & hist_reds)
        return total_match / len(recent)

    def random_individual():
        return tuple(sorted(_rand.sample(range(1, DLT_RED_COUNT+1), DLT_RED_PICK)))

    freq_pool = sorted(red_freq_all.keys(), key=lambda k: red_freq_all[k], reverse=True)
    population = []
    for _ in range(POP_SIZE):
        hot_k = min(15, len(freq_pool))
        hot   = _rand.sample(freq_pool[:hot_k], min(2, hot_k))
        rest  = _rand.sample([k for k in range(1, DLT_RED_COUNT+1) if k not in hot], DLT_RED_PICK - len(hot))
        population.append(tuple(sorted(hot + rest)))

    for gen in range(GENERATIONS):
        scored = sorted([(fitness(ind), ind) for ind in population], reverse=True)
        top = [ind for _, ind in scored[:POP_SIZE // 4]]
        new_pop = list(top)
        while len(new_pop) < POP_SIZE:
            p1, p2 = _rand.sample(top, 2)
            combined = list(set(p1) | set(p2))
            if len(combined) >= DLT_RED_PICK:
                child = tuple(sorted(_rand.sample(combined, DLT_RED_PICK)))
            else:
                child = random_individual()
            if _rand.random() < MUTATION_RATE:
                child = list(child)
                idx = _rand.randint(0, DLT_RED_PICK - 1)
                new_num = _rand.randint(1, DLT_RED_COUNT)
                while new_num in child:
                    new_num = _rand.randint(1, DLT_RED_COUNT)
                child[idx] = new_num
                child = tuple(sorted(child))
            new_pop.append(child)
        population = new_pop[:POP_SIZE]

    best = max(population, key=fitness)
    final_reds = list(best)

    sorted_blues = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)
    final_blues  = sorted(sorted_blues[:DLT_BLUE_PICK])

    hot_top  = sorted(red_freq_all.keys(),  key=lambda k: red_freq_all[k],  reverse=True)[:10]
    hot_btop = sorted(blue_freq_all.keys(), key=lambda k: blue_freq_all[k], reverse=True)[:5]

    return {
        "next_issue":   calc_next_issue(history[-1]["issue"]),
        "red_balls":    final_reds,
        "blue_balls":   final_blues,
        "method":       f"遗传算法（种群{POP_SIZE}，进化{GENERATIONS}代）",
        "detail":       {"说明": "将前区号码组合视为个体，用进化策略优化与历史开奖的吻合度",
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


def predict_next_issue(db_path, method="linear"):
    """
    大乐透主预测函数

    Args:
        db_path: 数据库路径
        method: "linear" / "frequency" / "entropy" / "gap" / "lstm" / "xgboost" / "gru"
                "hot_cold" / "same_period" / "consecutive" / "odd_even"
                "random_forest" / "bayesian" / "genetic"
    """
    history = load_history(db_path)
    if not history:
        return None

    if method == "frequency":
        return predict_frequency_weighted(history)
    elif method == "entropy":
        return predict_entropy_maximized(history)
    elif method == "gap":
        return predict_gap_maximized(history)
    elif method == "hot_cold":
        return predict_hot_cold_balance(history)
    elif method == "same_period":
        return predict_same_period_history(history)
    elif method == "consecutive":
        return predict_consecutive_span(history)
    elif method == "odd_even":
        return predict_odd_even_ratio(history)
    elif method == "random_forest":
        return predict_random_forest(db_path, history)
    elif method == "bayesian":
        return predict_bayesian(history)
    elif method == "genetic":
        return predict_genetic_algorithm(history)
    elif method == "lstm":
        # MLP 神经网络
        try:
            import lstm_dlt_model as lm
            result = lm.predict_lstm(db_path)
            if result is None:
                return {"issue": None, "red_balls": [], "blue_balls": [],
                        "method": "lstm", "error": "MLP 预测失败"}
            return result
        except Exception as e:
            return {"issue": None, "red_balls": [], "blue_balls": [],
                    "method": "lstm", "error": f"MLP模块加载失败: {e}"}
    elif method == "xgboost":
        # XGBoost 梯度提升
        try:
            import xgboost_dlt_model as xm
            result = xm.predict_xgboost(db_path)
            if result is None:
                return {"issue": None, "red_balls": [], "blue_balls": [],
                        "method": "xgboost", "error": "XGBoost 预测失败"}
            return result
        except Exception as e:
            return {"issue": None, "red_balls": [], "blue_balls": [],
                    "method": "xgboost", "error": f"XGBoost模块加载失败: {e}"}
    elif method == "gru":
        # GRU 序列模型
        try:
            import lstm_rnn_dlt_model as gm
            result = gm.predict_gru(db_path)
            if result is None:
                return {"issue": None, "red_balls": [], "blue_balls": [],
                        "method": "gru", "error": "GRU 预测失败"}
            return result
        except Exception as e:
            return {"issue": None, "red_balls": [], "blue_balls": [],
                    "method": "gru", "error": f"GRU模块加载失败: {e}"}
    else:
        return predict_linear_regression(history)


if __name__ == "__main__":
    import os
    db_path = os.path.join(os.path.dirname(__file__), "ssq.db")
    result = predict_next_issue(db_path, method="linear")
    if result:
        print(f"预测期号: {result['next_issue']}")
        print(f"前区号码: {result['red_balls']}")
        print(f"后区号码: {result['blue_balls']}")
    else:
        print("数据不足，请先抓取大乐透历史数据")
