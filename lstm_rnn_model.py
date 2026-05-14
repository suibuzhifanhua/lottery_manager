"""
双色球 LSTM/GRU 深度序列预测模块（PyTorch 实现）
──────────────────────────────────────────────
与 lstm_model.py（sklearn MLP）的区别：
  • 把每一期的原始号码作为时间步，送入 GRU 序列模型
  • GRU 能记住"前几期出过什么"，捕捉真正的时序依赖
  • 红球：33 维 sigmoid 多标签输出（每个球独立概率）
  • 蓝球：16 维 softmax 分类输出
"""
import os
import sys
import sqlite3
import numpy as np

# ─── 必须在 import torch 之前设置（禁用 dynamo 编译）────────────
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"
# 打包后可能没有 USER 环境变量，导致 getpass.getuser() 失败
if not os.environ.get("USER"):
    os.environ["USER"] = os.environ.get("USERNAME", "default_user")
if not os.environ.get("USERNAME"):
    os.environ["USERNAME"] = os.environ.get("USER", "default_user")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

from feature_utils import get_module_dir, load_history_from_db, calc_next_issue

# ── 配置 ────────────────────────────────────────────────────
RED_COUNT   = 33
BLUE_COUNT  = 16
SEQ_LEN     = 30          # 用最近 30 期作为序列输入
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3

MODEL_DIR       = os.path.join(get_module_dir(), "models")
GRU_MODEL_PATH  = os.path.join(MODEL_DIR, "ssq_gru_model.pt")
GRU_META_PATH   = os.path.join(MODEL_DIR, "ssq_gru_meta.pkl")   # 存归一化参数


# ── 模型定义 ─────────────────────────────────────────────────
class SSQGRUModel(nn.Module):
    """
    输入: (batch, seq_len, input_size)
    输出:
        red_logits:  (batch, 33)  — 过 sigmoid → 各红球出现概率
        blue_logits: (batch, 16)  — 过 softmax → 蓝球类别概率
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.red_head  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, RED_COUNT),
        )
        self.blue_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, BLUE_COUNT),
        )

    def forward(self, x):
        out, _ = self.gru(x)          # (batch, seq_len, hidden)
        last    = out[:, -1, :]       # 取最后一步
        return self.red_head(last), self.blue_head(last)


# ── 数据处理 ─────────────────────────────────────────────────
def load_history_from_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT issue, red1, red2, red3, red4, red5, red6, blue
        FROM ssq_history ORDER BY issue ASC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def encode_one_record(rec):
    """
    将一期数据编码为固定长度向量（49 维）
    红球 one-hot (33) + 蓝球 one-hot (16)
    """
    vec = np.zeros(RED_COUNT + BLUE_COUNT, dtype=np.float32)
    for i in range(1, 7):
        r = rec[i]
        if 1 <= r <= RED_COUNT:
            vec[r - 1] = 1.0
    b = rec[7]
    if 1 <= b <= BLUE_COUNT:
        vec[RED_COUNT + b - 1] = 1.0
    return vec


class SSQDataset(Dataset):
    def __init__(self, history, seq_len=SEQ_LEN):
        self.samples = []
        encoded = [encode_one_record(r) for r in history]
        for i in range(seq_len, len(history)):
            x = np.stack(encoded[i - seq_len: i])     # (seq_len, 49)
            # 目标：下一期的号码
            red_label  = np.zeros(RED_COUNT,  dtype=np.float32)
            blue_label = 0
            for j in range(1, 7):
                r = history[i][j]
                if 1 <= r <= RED_COUNT:
                    red_label[r - 1] = 1.0
            b = history[i][7]
            if 1 <= b <= BLUE_COUNT:
                blue_label = b - 1
            self.samples.append((x, red_label, blue_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, rl, bl = self.samples[idx]
        return torch.tensor(x), torch.tensor(rl), torch.tensor(bl, dtype=torch.long)


# ── 训练 ─────────────────────────────────────────────────────
def train_model(db_path):
    print("开始训练 GRU 序列模型...")
    print("=" * 50)

    history = load_history_from_db(db_path)
    if len(history) < SEQ_LEN + 20:
        print(f"数据不足: {len(history)} 期，需要至少 {SEQ_LEN + 20} 期")
        return None

    dataset    = SSQDataset(history)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = RED_COUNT + BLUE_COUNT   # 49
    model      = SSQGRUModel(input_size)
    device     = torch.device("cpu")
    model.to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    red_criterion = nn.BCEWithLogitsLoss()
    blue_criterion= nn.CrossEntropyLoss()

    print(f"样本数: {len(dataset)}, 输入维度: {input_size}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, rl, bl in dataloader:
            x, rl, bl = x.to(device), rl.to(device), bl.to(device)
            optimizer.zero_grad()
            red_logits, blue_logits = model(x)
            loss = red_criterion(red_logits, rl) + 0.5 * blue_criterion(blue_logits, bl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(dataloader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), GRU_MODEL_PATH)
    meta = {"input_size": input_size, "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS}
    joblib.dump(meta, GRU_META_PATH)
    print(f"模型已保存至 {GRU_MODEL_PATH}")
    return model


# ── 预测 ─────────────────────────────────────────────────────
def load_model():
    if not (os.path.exists(GRU_MODEL_PATH) and os.path.exists(GRU_META_PATH)):
        return None
    meta  = joblib.load(GRU_META_PATH)
    model = SSQGRUModel(meta["input_size"], meta["hidden_size"], meta["num_layers"])
    model.load_state_dict(torch.load(GRU_MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta


def predict_with_model(db_path):
    result = load_model()
    if result is None:
        print("模型不存在，开始训练...")
        model = train_model(db_path)
        if model is None:
            return None
        meta   = {"input_size": RED_COUNT + BLUE_COUNT, "seq_len": SEQ_LEN,
                  "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS}
    else:
        model, meta = result

    history = load_history_from_db(db_path)
    seq_len = meta["seq_len"]
    if len(history) < seq_len:
        print(f"历史数据不足 {seq_len} 期，无法预测")
        return None

    encoded = [encode_one_record(r) for r in history[-seq_len:]]
    x = torch.tensor(np.stack(encoded)).unsqueeze(0)   # (1, seq_len, 49)

    with torch.no_grad():
        red_logits, blue_logits = model(x)
        red_proba  = torch.sigmoid(red_logits).squeeze().numpy()
        blue_proba = torch.softmax(blue_logits, dim=-1).squeeze().numpy()

    top_red_indices = np.argsort(red_proba)[::-1][:6]
    top_reds        = sorted(int(i + 1) for i in top_red_indices)
    blue            = int(np.argmax(blue_proba)) + 1

    return {
        'red_balls':  top_reds,
        'blue_ball':  blue,
        'red_probs':  red_proba.tolist(),
        'blue_probs': blue_proba.tolist(),
    }


def predict_gru(db_path):
    """主预测函数，供 predict.py 调用，返回格式与 predict_lstm 一致"""
    try:
        result = predict_with_model(db_path)
        if result is None:
            return None

        history   = load_history_from_db(db_path)
        red_freq  = {i: 0 for i in range(1, 34)}
        blue_freq = {i: 0 for i in range(1, 17)}
        for rec in history:
            for i in range(1, 7):
                red_freq[rec[i]] += 1
            blue_freq[rec[7]] += 1

        hot_reds  = sorted(red_freq,  key=lambda k: red_freq[k],  reverse=True)[:10]
        hot_blues = sorted(blue_freq, key=lambda k: blue_freq[k], reverse=True)[:5]

        next_issue = calc_next_issue(history[-1][0])

        return {
            'next_issue':   next_issue,
            'red_balls':    result['red_balls'],
            'blue_ball':    result['blue_ball'],
            'method':       f'GRU序列学习（基于 {len(history)} 期，序列长度={SEQ_LEN}）',
            'detail': {
                '模型':     f'双层GRU（hidden={HIDDEN_SIZE}）',
                '输入':     f'最近{SEQ_LEN}期 one-hot 序列（{RED_COUNT+BLUE_COUNT}维/期）',
                '红球输出': '33维BCEWithLogits（多标签）',
                '蓝球输出': '16维CrossEntropy（分类）',
                '优化器':   f'Adam lr={LR}, epochs={EPOCHS}',
            },
            'red_freq':     red_freq,
            'blue_freq':    blue_freq,
            'hot_reds':     hot_reds,
            'hot_blues':    hot_blues,
            'history_used': len(history),
            'gru_result':   result,
        }

    except Exception as e:
        print(f"GRU预测出错: {e}")
        import traceback; traceback.print_exc()
        return None


if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "ssq.db")
    result  = predict_gru(db_path)
    if result:
        print(f"期号: {result['next_issue']}")
        print(f"红球: {result['red_balls']}")
        print(f"蓝球: {result['blue_ball']}")
        print(f"方法: {result['method']}")
    else:
        print("预测失败")
