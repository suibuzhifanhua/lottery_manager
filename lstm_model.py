"""
双色球 MLP 深度学习预测模块
使用 scikit-learn MLP 神经网络
输入：使用全部历史数据的统计特征
"""
import os
import sys
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 公共特征工程
from feature_utils import (
    get_module_dir, load_history_from_db, calc_next_issue, compute_features
)

# 配置
RED_COUNT  = 33
BLUE_COUNT = 16

MODEL_DIR   = os.path.join(get_module_dir(), "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "ssq_mlp_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "ssq_scaler.pkl")


def create_training_data(history):
    """创建训练数据"""
    X = []
    y_red = []
    y_blue = []
    
    # 至少需要10期历史才能预测下一期
    for i in range(10, len(history)):
        # 输入：前i期的特征
        X.append(compute_features(history[:i]))
        
        # 输出：第i期的红球和蓝球
        target_reds = history[i][1:7]
        target_blue = history[i][7]
        
        # 红球标签：one-hot
        red_label = np.zeros(RED_COUNT, dtype=np.float32)
        for r in target_reds:
            if 1 <= r <= RED_COUNT:
                red_label[r - 1] = 1
        y_red.append(red_label)
        
        # 蓝球标签
        blue_label = 0
        if 1 <= target_blue <= BLUE_COUNT:
            blue_label = target_blue - 1
        y_blue.append(blue_label)
    
    return np.array(X), np.array(y_red), np.array(y_blue)


def train_model(db_path):
    """训练 MLP 模型"""
    print("开始训练 MLP 神经网络...")
    print("=" * 50)
    
    history = load_history_from_db(db_path)
    if len(history) < 20:
        print(f"数据不足: {len(history)} 期，需要至少 20 期")
        return None, None
    
    X, y_red, y_blue = create_training_data(history)
    
    print(f"训练数据: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 红球模型 - 多标签分类
    print("训练红球模型...")
    red_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    red_model.fit(X_scaled, y_red)
    
    # 蓝球模型
    print("训练蓝球模型...")
    blue_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    blue_model.fit(X_scaled, y_blue)
    
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'red': red_model, 'blue': blue_model}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
    
    return {'red': red_model, 'blue': blue_model}, scaler


def load_model():
    """加载模型"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        models = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return models, scaler
    return None, None


def predict_with_model(db_path):
    """使用模型预测"""
    models, scaler = load_model()
    
    if models is None:
        print("模型不存在，开始训练...")
        models, scaler = train_model(db_path)
        if models is None:
            return None
    
    history = load_history_from_db(db_path)
    if len(history) < 10:
        print(f"数据不足: {len(history)} 期")
        return None
    
    # 用全部历史数据计算特征
    X = compute_features(history)
    X_scaled = scaler.transform(X.reshape(1, -1))
    
    # 预测红球概率
    red_proba = models['red'].predict_proba(X_scaled)[0]
    
    # 预测蓝球
    blue_pred = models['blue'].predict(X_scaled)[0]
    blue_proba = np.zeros(BLUE_COUNT)
    blue_proba[blue_pred] = 1.0
    
    # 选择概率最高的6个红球
    top_red_indices = np.argsort(red_proba)[::-1][:6]
    top_reds = sorted([int(i + 1) for i in top_red_indices])
    
    # 蓝球
    blue = int(blue_pred) + 1
    
    return {
        'red_balls': top_reds,
        'blue_ball': blue,
        'red_probs': red_proba.tolist(),
        'blue_probs': blue_proba.tolist()
    }


def incremental_train(db_path):
    """增量训练（重新训练）"""
    return train_model(db_path)


def predict_lstm(db_path):
    """
    主预测函数 - 供 predict.py 调用
    """
    try:
        result = predict_with_model(db_path)
        
        if result is None:
            return None
        
        # 计算频率信息
        history = load_history_from_db(db_path)
        red_freq = {i: 0 for i in range(1, 34)}
        blue_freq = {i: 0 for i in range(1, 17)}
        
        for rec in history:
            for i in range(1, 7):
                red_freq[rec[i]] += 1
            blue_freq[rec[7]] += 1
        
        hot_reds = sorted(red_freq.keys(), key=lambda k: red_freq[k], reverse=True)[:10]
        hot_blues = sorted(blue_freq.keys(), key=lambda k: blue_freq[k], reverse=True)[:5]
        
        # 计算下一期号
        next_issue = calc_next_issue(history[-1][0])
        
        return {
            'next_issue': next_issue,
            'red_balls': result['red_balls'],
            'blue_ball': result['blue_ball'],
            'method': f'MLP深度学习（基于 {len(history)} 期全部历史数据）',
            'detail': {
                '模型': 'MLP神经网络',
                '特征': '频率+遗漏+奇偶+大小+尾数+热号',
                '输入维度': 109,
                '红球输出': '33维sigmoid',
                '蓝球输出': '16维分类',
                '训练方式': '增量训练' if os.path.exists(MODEL_PATH) else '首次训练'
            },
            'red_freq': red_freq,
            'blue_freq': blue_freq,
            'hot_reds': hot_reds,
            'hot_blues': hot_blues,
            'history_used': len(history),
            'lstm_result': result
        }
        
    except Exception as e:
        print(f"MLP预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 测试
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "ssq.db")
    
    print("测试 MLP 神经网络预测（全部历史数据）")
    print("=" * 50)
    
    # 删除旧模型，重新训练
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("已删除旧模型")
    
    # 预测
    result = predict_lstm(db_path)
    
    if result:
        print(f"\n预测期号: {result['next_issue']}")
        print(f"红球: {result['red_balls']}")
        print(f"蓝球: {result['blue_ball']}")
        print(f"方法: {result['method']}")
        
        if 'lstm_result' in result:
            lstm = result['lstm_result']
            top_reds = np.argsort(lstm['red_probs'])[::-1][:6]
            red_info = [f"{i+1}({lstm['red_probs'][i]:.2%})" for i in top_reds]
            print(f"\n红球概率 Top 6: {red_info}")
            
            top_blue = np.argsort(lstm['blue_probs'])[::-1][:3]
            blue_info = [f"{i+1}({lstm['blue_probs'][i]:.2%})" for i in top_blue]
            print(f"蓝球概率 Top 3: {blue_info}")
    else:
        print("预测失败")
