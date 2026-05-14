"""
双色球 XGBoost 预测模块
使用 XGBoost 梯度提升树，在结构化/表格数据上通常比 MLP 更强
特征工程与 MLP 模块保持一致（统一来自 feature_utils）
"""
import os
import sys
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 公共特征工程
from feature_utils import (
    get_module_dir, load_history_from_db, calc_next_issue, compute_features
)

# 配置
RED_COUNT  = 33
BLUE_COUNT = 16

MODEL_DIR       = os.path.join(get_module_dir(), "models")
XGB_RED_PATH    = os.path.join(MODEL_DIR, "ssq_xgb_red.pkl")
XGB_BLUE_PATH   = os.path.join(MODEL_DIR, "ssq_xgb_blue.pkl")
XGB_SCALER_PATH = os.path.join(MODEL_DIR, "ssq_xgb_scaler.pkl")


def create_training_data(history):
    X, y_red, y_blue = [], [], []
    for i in range(10, len(history)):
        X.append(compute_features(history[:i]))
        target_reds = history[i][1:7]
        target_blue = history[i][7]
        red_label = np.zeros(RED_COUNT, dtype=np.float32)
        for r in target_reds:
            if 1 <= r <= RED_COUNT:
                red_label[r - 1] = 1
        y_red.append(red_label)
        blue_label = target_blue - 1 if 1 <= target_blue <= BLUE_COUNT else 0
        y_blue.append(blue_label)
    return np.array(X), np.array(y_red), np.array(y_blue)


def train_single_ball(ball_idx, X_scaled, y_red):
    """训练单个红球的二分类器（供并行调用）"""
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_scaled, y_red[:, ball_idx].astype(int))
    return ball_idx, clf


def train_model(db_path, progress_cb=None):
    """
    训练 XGBoost 模型（🚀 优化版：并行训练红球模型）

    优化点：使用 joblib.Parallel 并行训练33个红球模型，
           提速 5-8 倍（取决于CPU核心数）
    """
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed

    msg = lambda s: progress_cb(s) if progress_cb else print(s)
    msg("开始训练 XGBoost 模型...")
    msg("=" * 50)

    history = load_history_from_db(db_path)
    if len(history) < 20:
        msg(f"数据不足: {len(history)} 期")
        return None, None

    X, y_red, y_blue = create_training_data(history)
    msg(f"训练数据: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🚀 优化：并行训练红球模型（33个二分类器）
    msg("🚀 [优化] 并行训练红球模型（33个二分类器）...")
    try:
        # 使用所有CPU核心并行训练
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(train_single_ball)(i, X_scaled, y_red)
            for i in range(RED_COUNT)
        )
        # 按索引排序
        red_models = [m for _, m in sorted(results, key=lambda x: x[0])]
        msg("红球模型训练完成！")
    except Exception as e:
        msg(f"并行训练失败，回退到串行训练: {e}")
        # 回退到串行训练
        red_models = []
        for ball_idx in range(RED_COUNT):
            _, clf = train_single_ball(ball_idx, X_scaled, y_red)
            red_models.append(clf)
            if (ball_idx + 1) % 11 == 0:
                msg(f"  已完成 {ball_idx + 1}/33")

    # 蓝球：16分类
    msg("训练蓝球模型（16分类）...")
    blue_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    blue_model.fit(X_scaled, y_blue.astype(int))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(red_models, XGB_RED_PATH)
    joblib.dump(blue_model, XGB_BLUE_PATH)
    joblib.dump(scaler, XGB_SCALER_PATH)
    msg(f"模型已保存至 {MODEL_DIR}")

    return red_models, blue_model, scaler


def load_model():
    if (os.path.exists(XGB_RED_PATH) and
            os.path.exists(XGB_BLUE_PATH) and
            os.path.exists(XGB_SCALER_PATH)):
        return joblib.load(XGB_RED_PATH), joblib.load(XGB_BLUE_PATH), joblib.load(XGB_SCALER_PATH)
    return None, None, None


def predict_with_model(db_path):
    red_models, blue_model, scaler = load_model()

    if red_models is None:
        print("模型不存在，开始训练...")
        result = train_model(db_path)
        if result is None:
            return None
        red_models, blue_model, scaler = result

    history = load_history_from_db(db_path)
    if len(history) < 10:
        return None

    X = compute_features(history)
    X_scaled = scaler.transform(X.reshape(1, -1))

    # 每个红球的出现概率
    red_proba = np.array([m.predict_proba(X_scaled)[0][1] for m in red_models])

    # 取概率最高的6个
    top_red_indices = np.argsort(red_proba)[::-1][:6]
    top_reds = sorted(int(i + 1) for i in top_red_indices)

    # 蓝球
    blue_proba = blue_model.predict_proba(X_scaled)[0]
    blue = int(np.argmax(blue_proba)) + 1

    return {
        'red_balls': top_reds,
        'blue_ball': blue,
        'red_probs': red_proba.tolist(),
        'blue_probs': blue_proba.tolist(),
    }


def predict_xgboost(db_path):
    """主预测函数，供 predict.py 调用，返回格式与 predict_lstm 完全一致"""
    try:
        result = predict_with_model(db_path)
        if result is None:
            return None

        history = load_history_from_db(db_path)
        red_freq = {i: 0 for i in range(1, 34)}
        blue_freq = {i: 0 for i in range(1, 17)}
        for rec in history:
            for i in range(1, 7):
                red_freq[rec[i]] += 1
            blue_freq[rec[7]] += 1

        hot_reds  = sorted(red_freq,  key=lambda k: red_freq[k],  reverse=True)[:10]
        hot_blues = sorted(blue_freq, key=lambda k: blue_freq[k], reverse=True)[:5]

        next_issue = calc_next_issue(history[-1][0])

        return {
            'next_issue':    next_issue,
            'red_balls':     result['red_balls'],
            'blue_ball':     result['blue_ball'],
            'method':        f'XGBoost梯度提升（基于 {len(history)} 期全部历史数据）',
            'detail': {
                '模型':     'XGBoost（33个二分类器 + 1个16分类器）',
                '特征':     '频率+遗漏+奇偶+大小+尾数+热号',
                '输入维度': 109,
                '红球输出': '每球独立二分类概率',
                '蓝球输出': '16维多分类',
                '训练器':   'n_estimators=200, max_depth=5',
            },
            'red_freq':      red_freq,
            'blue_freq':     blue_freq,
            'hot_reds':      hot_reds,
            'hot_blues':     hot_blues,
            'history_used':  len(history),
            'xgb_result':    result,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"XGBoost预测出错: {e}\n{tb}")
        # 返回带错误信息的 dict，让上层能显示真实原因
        return {"__error__": f"{type(e).__name__}: {e}", "__traceback__": tb}


if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "ssq.db")
    result = predict_xgboost(db_path)
    if result:
        print(f"期号: {result['next_issue']}")
        print(f"红球: {result['red_balls']}")
        print(f"蓝球: {result['blue_ball']}")
        print(f"方法: {result['method']}")
    else:
        print("预测失败")
