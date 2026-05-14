"""
双色球共享特征工程与工具函数
避免 lstm_model.py / xgboost_model.py / predict.py 之间的代码重复
"""
import os
import sys
import sqlite3
from collections import Counter
# numpy 仅在深度学习预测时需要，延迟导入
# import numpy as np

RED_COUNT  = 33
BLUE_COUNT = 16


# ─── 路径工具 ─────────────────────────────────────────────────
def get_module_dir():
    """获取模块所在目录（打包后指向 _MEIPASS）"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


# ─── 数据库读取 ────────────────────────────────────────────────
def load_history_from_db(db_path):
    """
    从 SQLite 加载历史数据（按期号升序）
    返回 list of tuple: (issue, red1..red6, blue)
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT issue, red1, red2, red3, red4, red5, red6, blue
            FROM ssq_history
            ORDER BY issue ASC
        """)
        rows = c.fetchall()
    finally:
        conn.close()
    return rows


def load_history_as_dict(db_path):
    """
    从 SQLite 加载历史数据，返回 list of dict
    兼容 predict.py 中的 load_history()
    """
    rows = load_history_from_db(db_path)
    result = []
    for row in rows:
        result.append({
            "issue": row[0],
            "red1": row[1], "red2": row[2], "red3": row[3],
            "red4": row[4], "red5": row[5], "red6": row[6],
            "blue": row[7]
        })
    return result


# ─── 期号推算 ──────────────────────────────────────────────────
def calc_next_issue(last_issue):
    """根据最后一期推算下一期号（支持跨年）"""
    try:
        year = int(str(last_issue)[:2])
        num  = int(str(last_issue)[2:])
        if num >= 160:
            year += 1
            num = 1
        else:
            num += 1
        return f"{year:02d}{num:03d}"
    except Exception:
        return str(int(last_issue) + 1)


# ─── 遗漏值高效计算（O(n) 单次反向扫描） ──────────────────────
def compute_gap_fast(history):
    """
    一次反向遍历计算所有号码的当前遗漏值，O(n) 复杂度。

    history: list of tuple (issue, red1..red6, blue)

    返回:
        red_gap  dict {1..33: gap}
        blue_gap dict {1..16: gap}
    """
    n = len(history)
    red_gap  = {i: n for i in range(1, RED_COUNT  + 1)}   # 默认：从未出现 = 全程遗漏
    blue_gap = {i: n for i in range(1, BLUE_COUNT + 1)}

    red_found  = set()
    blue_found = set()

    for offset, rec in enumerate(reversed(history)):
        gap = offset  # 距最新一期的间隔
        reds = set(rec[1:7])
        blue = rec[7]

        for r in reds:
            if r not in red_found:
                red_gap[r] = gap
                red_found.add(r)

        if blue not in blue_found:
            blue_gap[blue] = gap
            blue_found.add(blue)

        # 全部号码都找到后提前退出
        if len(red_found) == RED_COUNT and len(blue_found) == BLUE_COUNT:
            break

    return red_gap, blue_gap


# ─── 109 维特征向量 ────────────────────────────────────────────
def compute_features(history, window=None):
    """
    计算 109 维特征向量（lstm_model 与 xgboost_model 共用）

    history : list of tuple (issue, red1..red6, blue)
    window  : 可选，限制只使用最近 window 期统计频率

    特征结构:
      红球频率    33 维
      蓝球频率    16 维
      红球遗漏值  33 维
      蓝球遗漏值  16 维
      奇偶比      2 维
      大小比      2 维
      尾数分布    10 维
      最近热号    10 维（one-hot，但只取 top10 对应位）
        注：原实现 shape 为 33 维 one-hot；
            这里保持与原版一致，输出 33 维 one-hot
      总期数      1 维
    """
    features = []

    use_history = history[-window:] if (window and len(history) > window) else history

    # 1. 红球频率 (33维)
    red_counter  = Counter()
    blue_counter = Counter()
    for rec in use_history:
        for i in range(1, 7):
            red_counter[rec[i]] += 1
        blue_counter[rec[7]] += 1

    total_reds = len(use_history) * 6
    for i in range(1, 34):
        features.append(red_counter.get(i, 0) / total_reds)

    # 2. 蓝球频率 (16维)
    total_blue = len(use_history)
    for i in range(1, 17):
        features.append(blue_counter.get(i, 0) / total_blue)

    # 3 & 4. 遗漏值 — 使用 O(n) 单次反向扫描
    red_gap, blue_gap = compute_gap_fast(use_history)
    for i in range(1, 34):
        features.append(red_gap[i])
    for i in range(1, 17):
        features.append(blue_gap[i])

    # 5. 红球奇偶比 (2维)
    odd_count = sum(1 for r in red_counter if r % 2 == 1)
    features.append(odd_count / 33)
    features.append((33 - odd_count) / 33)

    # 6. 红球大小比 (2维)  1-17 小 / 18-33 大
    small_count = sum(1 for r in red_counter if r <= 17)
    features.append(small_count / 33)
    features.append((33 - small_count) / 33)

    # 7. 尾数分布 (10维)
    tail_counter = Counter(r % 10 for r in red_counter)
    for i in range(10):
        features.append(tail_counter.get(i, 0) / 33)

    # 8. 最近30期热号 one-hot (33维) — 与原版保持一致
    recent = history[-30:] if len(history) > 30 else history
    recent_red = Counter()
    for rec in recent:
        for i in range(1, 7):
            recent_red[rec[i]] += 1
    hot_set = set(x[0] for x in sorted(recent_red.items(), key=lambda x: x[1], reverse=True)[:10])
    for i in range(1, 34):
        features.append(1 if i in hot_set else 0)

    # 9. 总期数归一化 (1维)
    features.append(len(use_history) / 5000)

    import numpy as np
    return np.array(features, dtype=np.float32)


# ─── 频率统计 ──────────────────────────────────────────────────
def calc_frequency(history_rows):
    """
    计算红球/蓝球历史频率
    history_rows: list of tuple (issue, red1..red6, blue)
    返回 (red_freq dict {1..33: count}, blue_freq dict {1..16: count})
    """
    red_freq  = {i: 0 for i in range(1, 34)}
    blue_freq = {i: 0 for i in range(1, 17)}
    for rec in history_rows:
        for i in range(1, 7):
            red_freq[rec[i]] += 1
        blue_freq[rec[7]] += 1
    return red_freq, blue_freq
