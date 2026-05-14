"""
购彩账本模块 v2.0
支持：双色球（SSQ）+ 大乐透（DLT）
规则：每次购买5注，每注2元，共10元/次
"""
import sqlite3
from datetime import datetime

# ─── 双色球奖级判定 ───────────────────────────────────────────
def check_prize_ssq(bought_reds, bought_blue, draw_reds, draw_blue):
    """
    判断一注双色球中奖情况
    bought_reds: list[int]  购买的6个红球
    bought_blue: int        购买的蓝球
    draw_reds:   list[int]  开奖红球
    draw_blue:   int        开奖蓝球
    返回 (prize_level: int, prize_name: str, prize_ref: str)
        prize_level: 0=未中奖, 1~6=奖级
    """
    red_match  = len(set(bought_reds) & set(draw_reds))
    blue_match = (bought_blue == draw_blue)

    if red_match == 6 and blue_match:
        return 1, "一等奖", "≈500万~数亿（浮动）"
    elif red_match == 6 and not blue_match:
        return 2, "二等奖", "≈5万~数十万（浮动）"
    elif red_match == 5 and blue_match:
        return 3, "三等奖", "3,000元"
    elif red_match == 5 and not blue_match:
        return 4, "四等奖", "200元"
    elif red_match == 4 and blue_match:
        return 4, "四等奖", "200元"
    elif red_match == 4 and not blue_match:
        return 5, "五等奖", "10元"
    elif red_match == 3 and blue_match:
        return 5, "五等奖", "10元"
    elif red_match == 2 and blue_match:
        return 6, "六等奖", "5元"
    elif red_match == 1 and blue_match:
        return 6, "六等奖", "5元"
    elif red_match == 0 and blue_match:
        return 6, "六等奖", "5元"
    else:
        return 0, "未中奖", "0元"


# ─── 大乐透奖级判定 ───────────────────────────────────────────
def check_prize_dlt(bought_front, bought_back, draw_front, draw_back):
    """
    判断一注大乐透中奖情况
    bought_front: list[int]  购买的前区5个号（1-35）
    bought_back:  list[int]  购买的后区2个号（1-12）
    draw_front:   list[int]  开奖前区5个号
    draw_back:   list[int]  开奖后区2个号
    返回 (prize_level: int, prize_name: str, prize_ref: str)
        浮动奖（一/二等奖）：按实际派奖
        固定奖（三~九等奖）：固定金额
    """
    front_hit = len(set(bought_front) & set(draw_front))
    back_hit  = len(set(bought_back)  & set(draw_back))

    # 一等奖：5+2
    if front_hit == 5 and back_hit == 2:
        return 1, "一等奖", "≈500万~数亿（浮动）"
    # 二等奖：5+1
    elif front_hit == 5 and back_hit == 1:
        return 2, "二等奖", "≈100万~数千万（浮动）"
    # 三等奖：5+0 或 4+2
    elif front_hit == 5 and back_hit == 0:
        return 3, "三等奖", "10,000元"
    elif front_hit == 4 and back_hit == 2:
        return 3, "三等奖", "10,000元"
    # 四等奖：4+1 或 3+2
    elif front_hit == 4 and back_hit == 1:
        return 4, "四等奖", "300元"
    elif front_hit == 3 and back_hit == 2:
        return 4, "四等奖", "300元"
    # 五等奖：4+0 或 3+1 或 2+2
    elif front_hit == 4 and back_hit == 0:
        return 5, "五等奖", "100元"
    elif front_hit == 3 and back_hit == 1:
        return 5, "五等奖", "100元"
    elif front_hit == 2 and back_hit == 2:
        return 5, "五等奖", "100元"
    # 六等奖：3+0 或 2+1 或 1+2 或 0+2
    elif front_hit == 3 and back_hit == 0:
        return 6, "六等奖", "15元"
    elif front_hit == 2 and back_hit == 1:
        return 6, "六等奖", "15元"
    elif front_hit == 1 and back_hit == 2:
        return 6, "六等奖", "15元"
    elif front_hit == 0 and back_hit == 2:
        return 6, "六等奖", "15元"
    # 七等奖：2+0 或 1+1 或 0+1
    elif front_hit == 2 and back_hit == 0:
        return 7, "七等奖", "5元"
    elif front_hit == 1 and back_hit == 1:
        return 7, "七等奖", "5元"
    elif front_hit == 0 and back_hit == 1:
        return 7, "七等奖", "5元"
    else:
        return 0, "未中奖", "0元"


def init_ledger_db(db_path):
    """初始化账本相关数据表（支持 SSQ + DLT）"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 购买批次表（含彩票类型）
    c.execute("""
    CREATE TABLE IF NOT EXISTS purchase_batch (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        lottery_type  TEXT NOT NULL DEFAULT 'ssq',   -- 'ssq' 或 'dlt'
        buy_date      TEXT NOT NULL,                   -- 购买日期 YYYY-MM-DD
        target_issue  TEXT NOT NULL,                   -- 目标期号
        notes         TEXT DEFAULT '',
        bet_count     INTEGER DEFAULT 5,               -- 注数（默认5注）
        unit_price    REAL DEFAULT 2.0,                -- 每注金额（元）
        is_extra      INTEGER DEFAULT 0,              -- 追加（仅大乐透有效）：0=否 1=是
        checked       INTEGER DEFAULT 0,               -- 是否已核对开奖结果(0/1)
        draw_issue    TEXT DEFAULT '',                 -- 实际开奖期号
        created_at    TEXT DEFAULT (datetime('now','localtime'))
    )
    """)

    # 单注购买记录表（兼容两种彩票）
    # SSQ: red1~red6 + blue,  blue2 = NULL
    # DLT: red1~red5 + blue + blue2, red6 = NULL
    c.execute("""
    CREATE TABLE IF NOT EXISTS purchase_ticket (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id      INTEGER NOT NULL,                -- 所属批次
        ticket_no     INTEGER NOT NULL,                -- 注号（1~5）
        red1 INTEGER, red2 INTEGER, red3 INTEGER,
        red4 INTEGER, red5 INTEGER, red6 INTEGER,     -- red6: DLT时为NULL
        blue  INTEGER,                                 -- SSQ蓝球 / DLT后区1
        blue2 INTEGER,                                 -- DLT后区2，SSQ时为NULL
        prize_level   INTEGER DEFAULT -1,              -- -1=未核对 0=未中奖 1~9=奖级
        prize_name    TEXT DEFAULT '',
        prize_ref     TEXT DEFAULT '',
        front_hit     INTEGER DEFAULT 0,              -- 前区命中数（DLT）
        back_hit      INTEGER DEFAULT 0,               -- 后区命中数（DLT）
        red_match     INTEGER DEFAULT 0,               -- 红球命中数（SSQ）
        blue_match    INTEGER DEFAULT 0,               -- 蓝球是否命中（SSQ, 0/1）
        FOREIGN KEY(batch_id) REFERENCES purchase_batch(id)
    )
    """)

    # 迁移旧数据：给已有记录补 lottery_type = 'ssq'
    # 注意：SQLite 的 ALTER TABLE ADD COLUMN 不支持 NOT NULL，只能用 DEFAULT
    try:
        c.execute("ALTER TABLE purchase_batch ADD COLUMN lottery_type TEXT DEFAULT 'ssq'")
    except Exception:
        pass

    # 迁移旧数据：给 purchase_batch 补 is_extra 列（旧库可能没有）
    try:
        c.execute("ALTER TABLE purchase_batch ADD COLUMN is_extra INTEGER DEFAULT 0")
    except Exception:
        pass

    # 迁移旧数据：给 purchase_ticket 补 blue2 列（旧库可能没有）
    try:
        c.execute("ALTER TABLE purchase_ticket ADD COLUMN blue2 INTEGER")
    except Exception:
        pass

    # 迁移旧数据：给 purchase_ticket 补中奖统计列
    for col_def in [
        "prize_level INTEGER DEFAULT -1",
        "prize_name TEXT DEFAULT ''",
        "prize_ref TEXT DEFAULT ''",
        "front_hit INTEGER DEFAULT 0",
        "back_hit INTEGER DEFAULT 0",
        "red_match INTEGER DEFAULT 0",
        "blue_match INTEGER DEFAULT 0",
    ]:
        col_name = col_def.split()[0]
        try:
            c.execute(f"ALTER TABLE purchase_ticket ADD COLUMN {col_def}")
        except Exception:
            pass

    conn.commit()
    conn.close()


# ─── 批次操作 ────────────────────────────────────────────────

def add_batch(db_path, lottery_type, buy_date, target_issue, tickets, notes="", bet_count=5, is_extra=False):
    """
    新增一次购买批次及5注号码
    lottery_type: 'ssq' 或 'dlt'
    tickets: list of dict
        SSQ: {red1~red6, blue}
        DLT: {red1~red5, blue, blue2}
    is_extra: 追加（仅大乐透有效，每注+1元）
    返回 batch_id
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO purchase_batch (lottery_type, buy_date, target_issue, notes, bet_count, is_extra)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (lottery_type, buy_date, target_issue, notes, bet_count, 1 if is_extra else 0))
    batch_id = c.lastrowid

    for i, t in enumerate(tickets, 1):
        if lottery_type == "ssq":
            c.execute("""
                INSERT INTO purchase_ticket
                (batch_id, ticket_no, red1, red2, red3, red4, red5, red6, blue, blue2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (batch_id, i, t["red1"], t["red2"], t["red3"],
                  t["red4"], t["red5"], t["red6"], t["blue"]))
        else:  # dlt
            c.execute("""
                INSERT INTO purchase_ticket
                (batch_id, ticket_no, red1, red2, red3, red4, red5, red6, blue, blue2)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """, (batch_id, i, t["red1"], t["red2"], t["red3"],
                  t["red4"], t["red5"], t["blue"], t["blue2"]))

    conn.commit()
    conn.close()
    return batch_id


def check_batch_result(db_path, batch_id, draw_issue, draw_data):
    """
    核对某批次的开奖结果，更新每注中奖情况
    draw_data: dict
        SSQ: {reds: [6个], blue: int}
        DLT: {front: [5个], back: [2个]}
    返回 list of result tuples
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT lottery_type FROM purchase_batch WHERE id=?", (batch_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    lottery_type = row[0]

    c.execute("""
        SELECT id, ticket_no, red1, red2, red3, red4, red5, red6, blue, blue2
        FROM purchase_ticket WHERE batch_id=?
        ORDER BY ticket_no
    """, (batch_id,))
    tickets = c.fetchall()
    results = []

    for tid, tno, r1, r2, r3, r4, r5, r6, b, b2 in tickets:
        if lottery_type == "ssq":
            bought_reds = [r1, r2, r3, r4, r5, r6]
            bought_blue = b
            lvl, name, ref = check_prize_ssq(
                bought_reds, bought_blue,
                draw_data["reds"], draw_data["blue"])
            red_match  = len(set(bought_reds) & set(draw_data["reds"]))
            blue_match = 1 if bought_blue == draw_data["blue"] else 0
            front_hit = back_hit = 0
        else:  # dlt
            bought_front = [r1, r2, r3, r4, r5]
            bought_back  = [b, b2]
            lvl, name, ref = check_prize_dlt(
                bought_front, bought_back,
                draw_data["front"], draw_data["back"])
            front_hit = len(set(bought_front) & set(draw_data["front"]))
            back_hit  = len(set(bought_back)  & set(draw_data["back"]))
            red_match = blue_match = 0

        c.execute("""
            UPDATE purchase_ticket
            SET prize_level=?, prize_name=?, prize_ref=?,
                front_hit=?, back_hit=?, red_match=?, blue_match=?
            WHERE id=?
        """, (lvl, name, ref, front_hit, back_hit, red_match, blue_match, tid))

        results.append((tno, lvl, name, ref, red_match, blue_match, front_hit, back_hit,
                        r1, r2, r3, r4, r5, r6, b, b2))

    c.execute("""
        UPDATE purchase_batch SET checked=1, draw_issue=? WHERE id=?
    """, (draw_issue, batch_id))

    conn.commit()
    conn.close()
    return results


def get_all_batches(db_path, lottery_type=None):
    """获取所有购买批次（含统计），可按彩票类型过滤"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if lottery_type:
        c.execute(f"""
            SELECT
                b.id, b.lottery_type, b.buy_date, b.target_issue, b.bet_count,
                (b.bet_count * 2.0 + CASE WHEN b.lottery_type = 'dlt' AND b.is_extra = 1 THEN b.bet_count * 1.0 ELSE 0 END) AS total_cost,
                b.is_extra,
                b.checked, b.draw_issue, b.notes, b.created_at,
                COUNT(CASE WHEN t.prize_level > 0 THEN 1 END) AS win_count,
                MAX(CASE WHEN t.prize_level > 0 THEN t.prize_level ELSE NULL END) AS best_level
            FROM purchase_batch b
            LEFT JOIN purchase_ticket t ON t.batch_id = b.id
            WHERE b.lottery_type = ?
            GROUP BY b.id
            ORDER BY b.id DESC
        """, (lottery_type,))
    else:
        c.execute(f"""
            SELECT
                b.id, b.lottery_type, b.buy_date, b.target_issue, b.bet_count,
                (b.bet_count * 2.0 + CASE WHEN b.lottery_type = 'dlt' AND b.is_extra = 1 THEN b.bet_count * 1.0 ELSE 0 END) AS total_cost,
                b.is_extra,
                b.checked, b.draw_issue, b.notes, b.created_at,
                COUNT(CASE WHEN t.prize_level > 0 THEN 1 END) AS win_count,
                MAX(CASE WHEN t.prize_level > 0 THEN t.prize_level ELSE NULL END) AS best_level
            FROM purchase_batch b
            LEFT JOIN purchase_ticket t ON t.batch_id = b.id
            GROUP BY b.id
            ORDER BY b.id DESC
        """)
    rows = c.fetchall()
    conn.close()
    return rows


def get_batch_tickets(db_path, batch_id):
    """获取某批次的所有票"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT ticket_no, red1, red2, red3, red4, red5, red6, blue, blue2,
               prize_level, prize_name, prize_ref, front_hit, back_hit,
               red_match, blue_match
        FROM purchase_ticket WHERE batch_id=?
        ORDER BY ticket_no
    """, (batch_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_summary(db_path, lottery_type=None):
    """获取账本总体统计"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 构建 WHERE 子句：有 lottery_type 时用 WHERE，无则空字符串
    if lottery_type:
        batch_where  = f"WHERE lottery_type = '{lottery_type}'"
        batch_and    = f"WHERE lottery_type = '{lottery_type}' AND"
        ticket_where = f"WHERE b.lottery_type = '{lottery_type}'"
        ticket_and   = f"WHERE b.lottery_type = '{lottery_type}' AND"
    else:
        batch_where  = ""
        batch_and    = "WHERE"
        ticket_where = ""
        ticket_and   = "WHERE"

    c.execute(f"""SELECT SUM(
        bet_count * 2.0 +
        CASE WHEN lottery_type = 'dlt' AND is_extra = 1
             THEN bet_count * 1.0 ELSE 0 END
    ) FROM purchase_batch {batch_where}""")
    total_cost = c.fetchone()[0] or 0.0

    c.execute(f"""SELECT SUM(
        bet_count * 2.0 +
        CASE WHEN lottery_type = 'dlt' AND is_extra = 1
             THEN bet_count * 1.0 ELSE 0 END
    ) FROM purchase_batch {batch_and} checked=1""")
    checked_cost = c.fetchone()[0] or 0.0

    c.execute(f"""
        SELECT t.prize_level, t.prize_name, COUNT(*) as cnt
        FROM purchase_ticket t
        JOIN purchase_batch b ON b.id = t.batch_id
        {ticket_and} t.prize_level > 0
        GROUP BY t.prize_level, t.prize_name
        ORDER BY t.prize_level
    """)
    prize_counts = c.fetchall()

    c.execute(f"""
        SELECT COUNT(*) FROM purchase_ticket t
        JOIN purchase_batch b ON b.id = t.batch_id
        {ticket_and} t.prize_level > 0
    """)
    total_win_tickets = c.fetchone()[0] or 0

    c.execute(f"""
        SELECT COUNT(*) FROM purchase_ticket t
        JOIN purchase_batch b ON b.id = t.batch_id
        {ticket_where}
    """)
    total_tickets = c.fetchone()[0] or 0

    c.execute(f"SELECT COUNT(*) FROM purchase_batch {batch_where}")
    total_batches = c.fetchone()[0] or 0

    c.execute(f"SELECT COUNT(*) FROM purchase_batch {batch_and} checked=1")
    checked_batches = c.fetchone()[0] or 0

    conn.close()
    return {
        "total_cost":        total_cost,
        "checked_cost":      checked_cost,
        "total_batches":     total_batches,
        "checked_batches":   checked_batches,
        "total_tickets":     total_tickets,
        "total_win_tickets": total_win_tickets,
        "prize_counts":      prize_counts,
    }


def delete_batch(db_path, batch_id):
    """删除一个购买批次（含所有票）"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM purchase_ticket WHERE batch_id=?", (batch_id,))
    c.execute("DELETE FROM purchase_batch WHERE id=?", (batch_id,))
    conn.commit()
    conn.close()


def get_latest_draw(db_path, lottery_type="ssq"):
    """从历史表获取最新一期开奖结果"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if lottery_type == "ssq":
        c.execute("""
            SELECT issue, draw_date, red1, red2, red3, red4, red5, red6, blue
            FROM ssq_history ORDER BY issue DESC LIMIT 1
        """)
        row = c.fetchone()
        conn.close()
        if row:
            return ("ssq", row)
        return None
    else:
        c.execute("""
            SELECT issue, draw_date, red1, red2, red3, red4, red5,
                   blue1, blue2
            FROM dlt_history ORDER BY issue DESC LIMIT 1
        """)
        row = c.fetchone()
        conn.close()
        if row:
            return ("dlt", row)
        return None


def get_draw_by_issue(db_path, lottery_type, issue):
    """按期号查询开奖结果"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if lottery_type == "ssq":
        c.execute("""
            SELECT issue, draw_date, red1, red2, red3, red4, red5, red6, blue
            FROM ssq_history WHERE issue=?
        """, (issue,))
        row = c.fetchone()
        conn.close()
        if row:
            return ("ssq", row)
    else:
        c.execute("""
            SELECT issue, draw_date, red1, red2, red3, red4, red5,
                   blue1, blue2
            FROM dlt_history WHERE issue=?
        """, (issue,))
        row = c.fetchone()
        conn.close()
        if row:
            return ("dlt", row)
    return None


def get_next_issue(db_path, lottery_type="ssq"):
    """获取下一期期号（用于自动填入）"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if lottery_type == "ssq":
        c.execute("SELECT issue FROM ssq_history ORDER BY issue DESC LIMIT 1")
    else:
        c.execute("SELECT issue FROM dlt_history ORDER BY issue DESC LIMIT 1")
    r = c.fetchone()
    conn.close()
    if r:
        return str(int(r[0]) + 1).zfill(5)
    return None


# ─── 统计 ─────────────────────────────────────────────────────

def get_weekly_stats(db_path, lottery_type=None):
    """按自然周统计"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    where = f"WHERE b.lottery_type = '{lottery_type}' " if lottery_type else ""

    c.execute(f"""
        SELECT
            strftime('%Y', b.buy_date)  AS yr,
            strftime('%W', b.buy_date)  AS wk,
            MIN(b.buy_date)             AS week_start,
            MAX(b.buy_date)             AS week_end,
            SUM(b.bet_count * 2.0)      AS cost,
            COUNT(*)                      AS batches
        FROM purchase_batch b
        {where}
        GROUP BY yr, wk
        ORDER BY yr DESC, wk DESC
        LIMIT 52
    """)
    rows = c.fetchall()

    c.execute(f"""
        SELECT
            strftime('%Y', b.buy_date) AS yr,
            strftime('%W', b.buy_date) AS wk,
            COUNT(t.id) AS win_tickets
        FROM purchase_batch b
        LEFT JOIN purchase_ticket t ON t.batch_id = b.id AND t.prize_level > 0
        {where}
        GROUP BY yr, wk
    """)
    win_map = {(r[0], r[1]): (r[2] or 0) for r in c.fetchall()}
    conn.close()

    result = []
    for yr, wk, week_start, week_end, cost, batches in rows:
        result.append({
            "week_label":  f"{yr}-W{int(wk)+1:02d}",
            "week_start":  week_start,
            "week_end":    week_end,
            "cost":        cost or 0.0,
            "batches":     batches,
            "win_tickets": win_map.get((yr, wk), 0),
        })
    return result


def get_monthly_stats(db_path, lottery_type=None):
    """按自然月统计"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    where = f"WHERE b.lottery_type = '{lottery_type}' " if lottery_type else ""

    c.execute(f"""
        SELECT
            strftime('%Y-%m', b.buy_date) AS month_label,
            SUM(b.bet_count * 2.0 +
                CASE WHEN b.lottery_type = 'dlt' AND b.is_extra = 1
                     THEN b.bet_count * 1.0 ELSE 0 END) AS cost,
            COUNT(*)                        AS batches
        FROM purchase_batch b
        {where}
        GROUP BY month_label
        ORDER BY month_label DESC
        LIMIT 36
    """)
    rows = c.fetchall()

    c.execute(f"""
        SELECT
            strftime('%Y-%m', b.buy_date) AS month_label,
            COUNT(t.id)                   AS win_tickets
        FROM purchase_batch b
        LEFT JOIN purchase_ticket t ON t.batch_id = b.id AND t.prize_level > 0
        {where}
        GROUP BY month_label
    """)
    win_map = {r[0]: (r[1] or 0) for r in c.fetchall()}
    conn.close()

    return [
        {
            "month_label": m,
            "cost":        c or 0.0,
            "batches":     b,
            "win_tickets": win_map.get(m, 0),
        }
        for m, c, b in rows
    ]


def get_yearly_stats(db_path, lottery_type=None):
    """按年统计"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    where = f"WHERE b.lottery_type = '{lottery_type}' " if lottery_type else ""

    c.execute(f"""
        SELECT
            strftime('%Y', b.buy_date) AS year_label,
            SUM(b.bet_count * 2.0 +
                CASE WHEN b.lottery_type = 'dlt' AND b.is_extra = 1
                     THEN b.bet_count * 1.0 ELSE 0 END) AS cost,
            COUNT(*)                 AS batches
        FROM purchase_batch b
        {where}
        GROUP BY year_label
        ORDER BY year_label DESC
    """)
    rows = c.fetchall()

    c.execute(f"""
        SELECT
            strftime('%Y', b.buy_date) AS year_label,
            COUNT(t.id)                AS win_tickets
        FROM purchase_batch b
        LEFT JOIN purchase_ticket t ON t.batch_id = b.id AND t.prize_level > 0
        {where}
        GROUP BY year_label
    """)
    win_map = {r[0]: (r[1] or 0) for r in c.fetchall()}
    conn.close()

    return [
        {
            "year_label":  y,
            "cost":        c or 0.0,
            "batches":     b,
            "win_tickets": win_map.get(y, 0),
        }
        for y, c, b in rows
    ]
