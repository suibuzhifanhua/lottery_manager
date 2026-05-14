"""
双色球历史数据抓取模块
数据来源：500彩票网
"""
import requests
import sqlite3
import re
import time
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/ssq/history/",
}

DB_PATH = "ssq.db"


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ssq_history (
        issue        TEXT PRIMARY KEY,
        draw_date    TEXT,
        red1         INTEGER,
        red2         INTEGER,
        red3         INTEGER,
        red4         INTEGER,
        red5         INTEGER,
        red6         INTEGER,
        blue         INTEGER,
        jackpot      TEXT,
        prize1_count INTEGER,
        prize2_count INTEGER,
        sales        TEXT,
        created_at   TEXT DEFAULT (datetime('now','localtime'))
    )
    """)
    conn.commit()
    conn.close()


def fetch_page(start="03001", end="99999"):
    url = (
        f"https://datachart.500.com/ssq/history/newinc/history.php"
        f"?start={start}&end={end}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "gb2312"
        return resp.text
    except Exception as e:
        print(f"[WARN] fetch failed: {e}")
        return ""


def parse_html(html):
    """
    解析500彩票网双色球历史页面，返回记录列表
    行格式：<tr class="t_tr1"><!--<td>2</td>--><td>期号</td>
            <td class="t_cfont2">红1</td>...<td class="t_cfont2">红6</td>
            <td class="t_cfont4">蓝</td><td class="t_cfont4">&nbsp;</td>
            <td>奖池</td><td>一等注数</td><td>一等金额</td>
            <td>二等注数</td><td>二等金额</td><td>总销售额</td><td>日期</td>
    """
    records = []
    # 逐行匹配 t_tr1 行
    row_pat = re.compile(r'<tr class="t_tr1">(.*?)</tr>', re.DOTALL)
    td_pat  = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
    strip_tag = lambda s: re.sub(r'<[^>]+>', '', s).strip()

    for row_m in row_pat.finditer(html):
        row_html = row_m.group(1)
        # 去掉注释 <!--...-->
        row_html_clean = re.sub(r'<!--.*?-->', '', row_html, flags=re.DOTALL)
        tds = [strip_tag(t) for t in td_pat.findall(row_html_clean)]
        # 至少需要 16 个 td（期号+6红+1蓝+1空白+奖池+一注+一金+二注+二金+销售+日期）
        if len(tds) < 15:
            continue
        try:
            issue     = tds[0].strip()
            if not re.match(r'^\d{5}$', issue):
                continue
            red1, red2, red3, red4, red5, red6 = (int(tds[i]) for i in range(1, 7))
            blue      = int(tds[7])
            jackpot   = tds[9].replace(',', '').replace('&nbsp;', '').strip()
            prize1    = int(tds[10].replace(',', '')) if tds[10].replace(',','').isdigit() else 0
            # tds[11] = 一等奖金额（单注）
            prize2    = int(tds[12].replace(',', '')) if tds[12].replace(',','').isdigit() else 0
            # tds[13] = 二等奖金额（单注）
            sales     = tds[14].replace(',', '').strip() if len(tds) > 14 else ''
            draw_date = tds[15].strip() if len(tds) > 15 else ''

            records.append({
                "issue":        issue,
                "draw_date":    draw_date,
                "red1": red1, "red2": red2, "red3": red3,
                "red4": red4, "red5": red5, "red6": red6,
                "blue":         blue,
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
         r["red4"], r["red5"], r["red6"], r["blue"], r["jackpot"],
         r["prize1_count"], r["prize2_count"], r["sales"])
        for r in records
    ]
    
    try:
        # 批量插入（事务内，一次性提交）
        c.executemany("""
            INSERT OR IGNORE INTO ssq_history
            (issue,draw_date,red1,red2,red3,red4,red5,red6,blue,jackpot,prize1_count,prize2_count,sales)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, data)
        conn.commit()
        count = c.rowcount
    except Exception as e:
        print(f"[WARN] batch insert error: {e}")
        conn.rollback()
        count = 0
    finally:
        conn.close()
    
    return count


def get_latest_issue(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT MAX(issue) FROM ssq_history")
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def fetch_all(db_path=DB_PATH, progress_cb=None):
    """抓取全部历史数据"""
    init_db(db_path)
    if progress_cb:
        progress_cb("正在请求历史数据，请稍候...")
    html = fetch_page("03001", "99999")
    if not html:
        if progress_cb:
            progress_cb("网络请求失败，请检查网络连接")
        return 0
    if progress_cb:
        progress_cb("解析数据中...")
    records = parse_html(html)
    if progress_cb:
        progress_cb(f"解析到 {len(records)} 条记录，正在写入数据库...")
    count = save_records(records, db_path)
    if progress_cb:
        progress_cb(f"完成！新增 {count} 条，共解析 {len(records)} 条")
    return count


def fetch_recent(db_path=DB_PATH, progress_cb=None):
    """仅抓取最新数据（增量）"""
    init_db(db_path)
    latest = get_latest_issue(db_path)
    start = latest if latest else "03001"
    if progress_cb:
        progress_cb(f"从期号 {start} 开始增量更新...")
    html = fetch_page(start, "99999")
    if not html:
        if progress_cb:
            progress_cb("网络请求失败")
        return 0
    records = parse_html(html)
    count = save_records(records, db_path)
    if progress_cb:
        progress_cb(f"[OK] 完成！新增 {count} 条")
    return count


if __name__ == "__main__":
    fetch_all(progress_cb=print)
