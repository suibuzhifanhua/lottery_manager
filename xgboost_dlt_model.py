"""
大乐透 XGBoost 梯度提升预测模块
使用 XGBoost 进行多标签分类预测
规则：前区1-35选5个 + 后区1-12选2个
"""
import os
import sys
import numpy as np
import joblib
from xgboost import XGBClassifier

# 大乐透规则
RED_COUNT  = 35   # 前区：1-35，选5个
BLUE_COUNT = 12   # 后区：1-12，选2个

# 打包后路径解析
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BASE_DIR

# 数据库使用 APP_DIR（用户数据目录），模型使用 BASE_DIR（打包资源目录）
DB_PATH = os.path.join(APP_DIR, "ssq.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dlt_xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "dlt_xgb_scaler.pkl")


def _calc_next_issue(last_issue):
    """根据最后一期推算下一期号"""
    try:
        s = str(last_issue)
        if len(s) == 7:
            year = int(s[:4])
            num = int(s[4:])
        else:
            year = int(s[:2])
            num = int(s[2:])
        if num >= 180:
            year += 1
            num = 1
        else:
            num += 1
        if len(s) == 7:
            return f"{year:04d}{num:03d}"
        else:
            return f"{year:02d}{num:03d}"
    except:
        return str(int(last_issue) + 1)


def _compute_gap(history):
    """计算遗漏值"""
    n = len(history)
    red_gap = {i: n for i in range(1, RED_COUNT + 1)}
    blue_gap = {i: n for i in range(1, BLUE_COUNT + 1)}
    red_found = set()
    blue_found = set()
    
    for offset, rec in enumerate(reversed(history)):
        gap = offset
        reds = {rec[f"red{i}"] for i in range(1, 6)}
        blues = {rec[f"blue{i}"] for i in range(1, 3)}
        
        for r in reds:
            if r not in red_found:
                red_gap[r] = gap
                red_found.add(r)
        for b in blues:
            if b not in blue_found:
                blue_gap[b] = gap
                blue_found.add(b)
        
        if len(red_found) == RED_COUNT and len(blue_found) == BLUE_COUNT:
            break
    
    return red_gap, blue_gap


def compute_features(history):
    """计算大乐透特征向量"""
    n = len(history)
    if n < 10:
        return np.zeros(120)
    
    red_freq = {i: 0 for i in range(1, RED_COUNT + 1)}
    blue_freq = {i: 0 for i in range(1, BLUE_COUNT + 1)}
    
    for rec in history:
        for i in range(1, 6):
            red_freq[rec[f"red{i}"]] += 1
        for i in range(1, 3):
            blue_freq[rec[f"blue{i}"]] += 1
    
    # 频率特征
    red_freq_vals = [red_freq[i] / n for i in range(1, RED_COUNT + 1)]
    blue_freq_vals = [blue_freq[i] / n for i in range(1, BLUE_COUNT + 1)]
    
    # 遗漏值特征
    red_gap, blue_gap = _compute_gap(history)
    red_gap_vals = [red_gap[i] / n for i in range(1, RED_COUNT + 1)]
    blue_gap_vals = [blue_gap[i] / n for i in range(1, BLUE_COUNT + 1)]
    
    # 近期趋势（最近5期）
    recent_freq = {i: 0 for i in range(1, RED_COUNT + 1)}
    recent_blue_freq = {i: 0 for i in range(1, BLUE_COUNT + 1)}
    for rec in history[-5:]:
        for i in range(1, 6):
            recent_freq[rec[f"red{i}"]] += 1
        for i in range(1, 3):
            recent_blue_freq[rec[f"blue{i}"]] += 1
    
    recent_red_vals = [recent_freq[i] / 5 for i in range(1, RED_COUNT + 1)]
    recent_blue_vals = [recent_blue_freq[i] / 5 for i in range(1, BLUE_COUNT + 1)]
    
    features = (
        red_freq_vals + blue_freq_vals +
        red_gap_vals + blue_gap_vals +
        recent_red_vals + recent_blue_vals
    )
    
    return np.array(features, dtype=np.float32)


def create_training_data(history):
    """创建训练数据"""
    X = []
    y_red = []
    y_blue = []
    
    for i in range(10, len(history)):
        X.append(compute_features(history[:i]))
        
        # 输出：第i期的前区和后区
        target_reds = [history[i][f"red{j}"] for j in range(1, 6)]
        target_blues = [history[i][f"blue{j}"] for j in range(1, 3)]
        
        # 前区标签：one-hot
        red_label = np.zeros(RED_COUNT, dtype=np.float32)
        for r in target_reds:
            if 1 <= r <= RED_COUNT:
                red_label[r - 1] = 1
        y_red.append(red_label)
        
        # 后区标签
        blue_label = np.zeros(BLUE_COUNT, dtype=np.float32)
        for b in target_blues:
            if 1 <= b <= BLUE_COUNT:
                blue_label[b - 1] = 1
        y_blue.append(blue_label)
    
    return np.array(X), np.array(y_red), np.array(y_blue)


def load_dlt_history():
    """加载大乐透历史数据"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""
            SELECT issue, red1, red2, red3, red4, red5, blue1, blue2
            FROM dlt_history ORDER BY issue ASC
        """)
        rows = c.fetchall()
    finally:
        conn.close()
    return [{"issue": r[0], "red1": r[1], "red2": r[2], "red3": r[3], "red4": r[4], "red5": r[5],
             "blue1": r[6], "blue2": r[7]} for r in rows]


def train_model(db_path=None):
    """训练 XGBoost 模型"""
    if db_path is None:
        db_path = DB_PATH
    
    print("开始训练大乐透 XGBoost 模型...")
    print("=" * 50)
    
    history = load_dlt_history()
    if len(history) < 20:
        print(f"数据不足: {len(history)} 期，需要至少 20 期")
        return None, None
    
    X, y_red, y_blue = create_training_data(history)
    
    print(f"训练数据: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")
    
    # 前区模型
    print("训练前区模型...")
    red_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    red_model.fit(X, y_red)
    
    # 后区模型
    print("训练后区模型...")
    blue_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    blue_model.fit(X, y_blue)
    
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'red': red_model, 'blue': blue_model}, MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
    
    return {'red': red_model, 'blue': blue_model}


def load_model():
    """加载模型"""
    if os.path.exists(MODEL_PATH):
        models = joblib.load(MODEL_PATH)
        return models
    return None


def predict_with_model(db_path=None):
    """使用模型预测"""
    if db_path is None:
        db_path = DB_PATH
    
    models = load_model()
    
    if models is None:
        print("模型不存在，开始训练...")
        models = train_model(db_path)
        if models is None:
            return None
    
    history = load_dlt_history()
    if len(history) < 10:
        print(f"数据不足: {len(history)} 期")
        return None
    
    # 用全部历史数据计算特征
    X = compute_features(history)
    X_scaled = X.reshape(1, -1)
    
    # 预测前区概率
    red_proba = models['red'].predict_proba(X_scaled)[0]
    
    # 预测后区概率
    blue_proba = models['blue'].predict_proba(X_scaled)[0]
    
    # 选择概率最高的5个前区
    top_red_indices = np.argsort(red_proba)[::-1][:5]
    top_reds = sorted([int(i + 1) for i in top_red_indices])
    
    # 选择概率最高的2个后区
    top_blue_indices = np.argsort(blue_proba)[::-1][:2]
    top_blues = sorted([int(i + 1) for i in top_blue_indices])
    
    return {
        'red_balls': top_reds,
        'blue_balls': top_blues,
        'red_probs': red_proba.tolist(),
        'blue_probs': blue_proba.tolist()
    }


def predict_xgboost(db_path=None):
    """主预测函数 - 供 predict_dlt.py 调用"""
    try:
        result = predict_with_model(db_path)
        
        if result is None:
            return None
        
        history = load_dlt_history()
        red_freq = {i: 0 for i in range(1, RED_COUNT + 1)}
        blue_freq = {i: 0 for i in range(1, BLUE_COUNT + 1)}
        
        for rec in history:
            for i in range(1, 6):
                red_freq[rec[f"red{i}"]] += 1
            for i in range(1, 3):
                blue_freq[rec[f"blue{i}"]] += 1
        
        hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
        hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]
        
        next_issue = _calc_next_issue(history[-1]["issue"])
        
        return {
            'next_issue': next_issue,
            'red_balls': result['red_balls'],
            'blue_balls': result['blue_balls'],
            'method': f'XGBoost梯度提升（基于 {len(history)} 期全部历史数据）',
            'detail': {
                '模型': 'XGBoost',
                '特征': '频率+遗漏+近期趋势',
                '前区输出': '35维概率',
                '后区输出': '12维概率',
                '训练方式': '增量训练' if os.path.exists(MODEL_PATH) else '首次训练'
            },
            'red_freq': red_freq,
            'blue_freq': blue_freq,
            'hot_reds': hot_reds,
            'hot_blues': hot_blues,
            'history_used': len(history),
            'xgb_result': result
        }
        
    except Exception as e:
        print(f"XGBoost预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("测试大乐透 XGBoost 预测")
    print("=" * 50)
    
    result = predict_xgboost()
    
    if result:
        print(f"\n预测期号: {result['next_issue']}")
        print(f"前区: {result['red_balls']}")
        print(f"后区: {result['blue_balls']}")
        print(f"方法: {result['method']}")
    else:
        print("预测失败")
