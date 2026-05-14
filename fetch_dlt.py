"""
大乐透历史数据抓取模块
数据来源：500彩票网
大乐透规则：从1-35中选5个前区号 + 从1-12中选2个后区号
"""
import requests
import sqlite3
import re
import time
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/dlt/history/",
}

DB_PATH = "ssq.db"


def init_dlt_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS dlt_history (
        issue        TEXT PRIMARY KEY,
        draw_date    TEXT,
        red1         INTEGER,
        red2         INTEGER,
        red3         INTEGER,
        red4         INTEGER,
        red5         INTEGER,
        blue1        INTEGER,
        blue2        INTEGER,
        jackpot      TEXT,
        prize1_count INTEGER,
        prize2_count INTEGER,
        sales        TEXT,
        created_at   TEXT DEFAULT (datetime('now','localtime'))
    )
    """)
    conn.commit()
    conn.close()


def fetch_page(start="00001", end="99999"):
    url = (
        f"https://datachart.500.com/dlt/history/newinc/history.php"
        f"?start={start}&end={end}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "gb2312"
        return resp.text
    except Exception as e:
        print(f"[WARN] 大乐透 fetch failed: {e}")
        return ""


def parse_html(html):
    """
    解析500彩票网大乐透历史页面，返回记录列表
    大乐透行格式：期号 + 5个前区红球 + 2个后区蓝球 + 奖池 + 一等注数 + ...
    """
    records = []
    row_pat = re.compile(r'<tr class="t_tr1">(.*?)</tr>', re.DOTALL)
    td_pat  = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
    strip_tag = lambda s: re.sub(r'<[^>]+>', '', s).strip()

    for row_m in row_pat.finditer(html):
        row_html = row_m.group(1)
        row_html_clean = re.sub(r'<!--.*?-->', '', row_html, flags=re.DOTALL)
        tds = [strip_tag(t) for t in td_pat.findall(row_html_clean)]

        if len(tds) < 12:
            continue
        try:
            issue = tds[0].strip()
            # 大乐透期号格式：5位数字（如 25001）或 7位（如 2025001）
            if not re.match(r'^\d{5,7}$', issue):
                continue

            red1 = int(tds[1])
            red2 = int(tds[2])
            red3 = int(tds[3])
            red4 = int(tds[4])
            red5 = int(tds[5])
            blue1 = int(tds[6])
            blue2 = int(tds[7])

            # 验证号码合法范围
            if not all(1 <= r <= 35 for r in [red1, red2, red3, red4, red5]):
                continue
            if not all(1 <= b <= 12 for b in [blue1, blue2]):
                continue

            # 后续字段根据实际HTML位置解析（奖池、注数、销售额等）
            jackpot = ""
            prize1  = 0
            prize2  = 0
            sales   = ""
            draw_date = ""

            # 尝试从后续 tds 中提取这些字段
            for i in range(8, min(len(tds), 20)):
                val = tds[i].replace(',', '').replace('&nbsp;', '').strip()
                if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
                    draw_date = val
                elif re.match(r'^\d{6,}$', val) and not jackpot:
                    jackpot = val
                elif re.match(r'^\d{1,5}$', val) and prize1 == 0:
                    prize1 = int(val) if val else 0
                elif re.match(r'^\d{1,5}$', val) and prize2 == 0 and val != str(prize1):
                    prize2 = int(val) if val else 0

            records.append({
                "issue":        issue,
                "draw_date":    draw_date,
                "red1": red1, "red2": red2, "red3": red3,
                "red4": red4, "red5": red5,
                "blue1": blue1, "blue2": blue2,
                "jackpot":      jackpot,
                "prize1_count": prize1,
                "prize2_count": prize2,
                "sales":        sales,
            })
        except Exception:
            continue
    return records


def save_records(records, db_path=DB_PATH):
    """批量插入记录，比逐条插入快 20-50 倍"""
    if not records:
        return 0
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 批量数据
    data = [
        (r["issue"], r["draw_date"], r["red1"], r["red2"], r["red3"],
         r["red4"], r["red5"], r["blue1"], r["blue2"], r["jackpot"],
         r["prize1_count"], r["prize2_count"], r["sales"])
        for r in records
    ]
    
    try:
        # 批量插入（事务内，一次性提交）
        c.executemany("""
            INSERT OR IGNORE INTO dlt_history
            (issue,draw_date,red1,red2,red3,red4,red5,blue1,blue2,jackpot,prize1_count,prize2_count,sales)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, data)
        conn.commit()
        count = c.rowcount
    except Exception as e:
        print(f"[WARN] 大乐透批量插入 error: {e}")
        conn.rollback()
        count = 0
    finally:
        conn.close()
    
    return count


def get_latest_issue(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT MAX(issue) FROM dlt_history")
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def fetch_all(db_path=DB_PATH, progress_cb=None):
    """抓取全部大乐透历史数据"""
    init_dlt_db(db_path)
    if progress_cb:
        progress_cb("大乐透：正在请求历史数据，请稍候...")
    html = fetch_page("00001", "99999")
    if not html:
        if progress_cb:
            progress_cb("大乐透：网络请求失败，请检查网络连接")
        return 0
    if progress_cb:
        progress_cb("大乐透：解析数据中...")
    records = parse_html(html)
    if progress_cb:
        progress_cb(f"大乐透：解析到 {len(records)} 条记录，正在写入数据库...")
    count = save_records(records, db_path)
    if progress_cb:
        progress_cb(f"大乐透：完成！新增 {count} 条，共解析 {len(records)} 条")
    return count


def fetch_recent(db_path=DB_PATH, progress_cb=None):
    """仅抓取大乐透最新数据（增量）"""
    init_dlt_db(db_path)
    latest = get_latest_issue(db_path)
    start = latest if latest else "00001"
    if progress_cb:
        progress_cb(f"大乐透：从期号 {start} 开始增量更新...")
    html = fetch_page(start, "99999")
    if not html:
        if progress_cb:
            progress_cb("大乐透：网络请求失败")
        return 0
    records = parse_html(html)
    count = save_records(records, db_path)
    if progress_cb:
        progress_cb(f"大乐透：[OK] 完成！新增 {count} 条")
    return count


def count_records(db_path=DB_PATH):
    """获取大乐透数据库记录数"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM dlt_history")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


if __name__ == "__main__":
    fetch_all(progress_cb=print)
