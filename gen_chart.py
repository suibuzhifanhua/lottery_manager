"""
双色球号码分布图表生成模块（🚀 优化版）
支持从数据库读取数据，自动检测路径
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径配置（兼容打包后运行）──────────────────────────────
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
APP_DIR = get_app_dir()

# 优先从数据库读取，其次从Excel
DB_PATH = os.path.join(APP_DIR, "ssq.db")
EXCEL_PATH = os.path.join(APP_DIR, "双色球历史数据.xlsx")

# 加载数据
df = None
if os.path.exists(DB_PATH):
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT issue, draw_date, red1, red2, red3, red4, red5, red6, blue FROM ssq_history ORDER BY issue", conn)
        conn.close()
        if not df.empty:
            df.columns = ['期号', '开奖日期', '红球1', '红球2', '红球3', '红球4', '红球5', '红球6', '蓝球']
            print(f"[OK] 从数据库加载了 {len(df)} 条记录")
    except Exception as e:
        print(f"[WARN] 数据库读取失败: {e}")

if df is None or df.empty:
    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        print(f"[OK] 从Excel加载了 {len(df)} 条记录")
    else:
        print("[ERROR] 未找到数据文件，请先导入历史数据！")
        sys.exit(1)

total = len(df)

# 统计频率
red_cols = ['红球1','红球2','红球3','红球4','红球5','红球6']
red_counter = Counter()
for col in red_cols:
    red_counter.update(df[col].dropna().astype(int).tolist())
blue_counter = Counter(df['蓝球'].dropna().astype(int).tolist())

red_nums  = list(range(1, 34))
blue_nums = list(range(1, 17))
red_freq  = [red_counter.get(n, 0) for n in red_nums]
blue_freq = [blue_counter.get(n, 0) for n in blue_nums]

# 颜色：按频率深浅
def freq_colors(freqs, base_rgb):
    arr = np.array(freqs, dtype=float)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    colors = []
    for v in norm:
        r = base_rgb[0] + (1 - base_rgb[0]) * (1 - v) * 0.55
        g = base_rgb[1] + (1 - base_rgb[1]) * (1 - v) * 0.55
        b = base_rgb[2] + (1 - base_rgb[2]) * (1 - v) * 0.55
        colors.append((r, g, b))
    return colors

red_colors  = freq_colors(red_freq,  (0.85, 0.15, 0.15))
blue_colors = freq_colors(blue_freq, (0.15, 0.38, 0.82))

# 期望次数
expected_red  = total * 6 / 33
expected_blue = total / 16

# ── 画布 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')

gs = fig.add_gridspec(3, 2,
    left=0.06, right=0.97, top=0.90, bottom=0.06,
    hspace=0.52, wspace=0.28,
    height_ratios=[2.2, 2.2, 1.3])

ax_red_bar  = fig.add_subplot(gs[0, :])   # 红球频次柱状图（全宽）
ax_blue_bar = fig.add_subplot(gs[1, :])   # 蓝球频次柱状图（全宽）
ax_red_pie  = fig.add_subplot(gs[2, 0])   # 红球 Top10 饼图
ax_blue_pie = fig.add_subplot(gs[2, 1])   # 蓝球分布饼图

TITLE_C  = '#e0e0ff'
LABEL_C  = '#b0b8d8'
GRID_C   = '#2a2a4a'
BG_AX    = '#0f0f23'

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=LABEL_C, labelsize=9)
    ax.spines[:].set_color(GRID_C)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)

# ── 红球柱状图 ────────────────────────────────────────────────
style_ax(ax_red_bar)
bars_r = ax_red_bar.bar(red_nums, red_freq, color=red_colors,
                        width=0.72, zorder=3, edgecolor='none')
ax_red_bar.axhline(expected_red, color='#ffdd57', linewidth=1.4,
                   linestyle='--', label=f'期望均值 {expected_red:.0f}次', zorder=4)

# 标注最高 / 最低
max_r = max(red_freq); min_r = min(red_freq)
for i, (n, f) in enumerate(zip(red_nums, red_freq)):
    if f == max_r or f == min_r:
        clr = '#ff6b6b' if f == max_r else '#74c0fc'
        ax_red_bar.text(n, f + max_r * 0.012, str(f),
                        ha='center', va='bottom', fontsize=7.5,
                        color=clr, fontweight='bold')

ax_red_bar.set_title(f'红球出现频次分布（共 {total} 期，每期6个红球）',
                     color=TITLE_C, fontsize=13, fontweight='bold', pad=8)
ax_red_bar.set_xlabel('红球号码（1–33）', color=LABEL_C, fontsize=9)
ax_red_bar.set_ylabel('出现次数', color=LABEL_C, fontsize=9)
ax_red_bar.set_xticks(red_nums)
ax_red_bar.set_xticklabels(red_nums, fontsize=8)
leg = ax_red_bar.legend(facecolor='#1a1a2e', edgecolor=GRID_C,
                        labelcolor=LABEL_C, fontsize=9)

# ── 蓝球柱状图 ────────────────────────────────────────────────
style_ax(ax_blue_bar)
bars_b = ax_blue_bar.bar(blue_nums, blue_freq, color=blue_colors,
                         width=0.72, zorder=3, edgecolor='none')
ax_blue_bar.axhline(expected_blue, color='#ffdd57', linewidth=1.4,
                    linestyle='--', label=f'期望均值 {expected_blue:.0f}次', zorder=4)

max_b = max(blue_freq); min_b = min(blue_freq)
for i, (n, f) in enumerate(zip(blue_nums, blue_freq)):
    if f == max_b or f == min_b:
        clr = '#ff6b6b' if f == max_b else '#74c0fc'
        ax_blue_bar.text(n, f + max_b * 0.012, str(f),
                         ha='center', va='bottom', fontsize=8,
                         color=clr, fontweight='bold')

ax_blue_bar.set_title(f'蓝球出现频次分布（共 {total} 期，每期1个蓝球）',
                      color=TITLE_C, fontsize=13, fontweight='bold', pad=8)
ax_blue_bar.set_xlabel('蓝球号码（1–16）', color=LABEL_C, fontsize=9)
ax_blue_bar.set_ylabel('出现次数', color=LABEL_C, fontsize=9)
ax_blue_bar.set_xticks(blue_nums)
ax_blue_bar.legend(facecolor='#1a1a2e', edgecolor=GRID_C,
                   labelcolor=LABEL_C, fontsize=9)

# ── 红球 Top10 饼图 ───────────────────────────────────────────
ax_red_pie.set_facecolor(BG_AX)
top10_idx  = sorted(range(33), key=lambda i: red_freq[i], reverse=True)[:10]
top10_nums = [red_nums[i] for i in top10_idx]
top10_freq = [red_freq[i] for i in top10_idx]
pie_colors_r = [red_colors[i] for i in top10_idx]
wedges, texts, autotexts = ax_red_pie.pie(
    top10_freq, labels=[f'{n}号' for n in top10_nums],
    colors=pie_colors_r, autopct='%1.1f%%',
    startangle=90, pctdistance=0.78,
    textprops={'color': LABEL_C, 'fontsize': 8},
    wedgeprops={'linewidth': 0.5, 'edgecolor': '#1a1a2e'})
for at in autotexts:
    at.set_color('#ffffff')
    at.set_fontsize(7.5)
ax_red_pie.set_title('红球出现频次 Top 10',
                     color=TITLE_C, fontsize=11, fontweight='bold', pad=6)

# ── 蓝球饼图 ─────────────────────────────────────────────────
ax_blue_pie.set_facecolor(BG_AX)
wedges2, texts2, autotexts2 = ax_blue_pie.pie(
    blue_freq, labels=[f'{n}号' for n in blue_nums],
    colors=blue_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.78,
    textprops={'color': LABEL_C, 'fontsize': 8},
    wedgeprops={'linewidth': 0.5, 'edgecolor': '#1a1a2e'})
for at in autotexts2:
    at.set_color('#ffffff')
    at.set_fontsize(7.5)
ax_blue_pie.set_title('蓝球各号码出现比例',
                      color=TITLE_C, fontsize=11, fontweight='bold', pad=6)

# ── 总标题 ───────────────────────────────────────────────────
fig.suptitle('双色球历史数据  号码分布全景分析',
             color='#ffffff', fontsize=17, fontweight='bold', y=0.965)

# 副标题
fig.text(0.5, 0.932,
         f'数据范围：{df["期号"].iloc[-1]} ~ {df["期号"].iloc[0]}  |  共 {total} 期  |  颜色越深表示出现频率越高',
         ha='center', color='#8898cc', fontsize=9.5)

OUT = os.path.join(APP_DIR, "双色球号码分布.png")
plt.savefig(OUT, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'[OK] Chart saved: {OUT}')
