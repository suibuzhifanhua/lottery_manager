# 🎰 彩票管理器 (Lottery Manager)

> 彩票历史数据管理与预测系统，支持双色球（SSQ）和大乐透（DLT）。

![Version](https://img.shields.io/badge/version-v2.4.0-blue)
![Python](https://img.shields.io/badge/python-3.14-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 功能特性

### 历史数据管理
- **双色球/大乐透历史数据**查询、导入、导出
- **在线抓取**最新开奖数据
- **Excel 导出**功能

### 智能预测
- 🎯 **多种机器学习模型**：XGBoost、Random Forest、LSTM、RNN 等
- 📊 **特征工程**：基于历史规律生成特征
- ⚡ **并行优化**：多进程加速训练

### 账本管理
- 📒 购彩记录记账
- 💰 盈亏统计与分析
- 📈 历史中奖查询

## 🚀 快速开始

### 下载运行（Windows）

下载最新版本的可执行文件，双击即可运行，无需安装 Python 环境：

```
dist/lottery_manager_v2.4.0.exe
```

### 从源码运行

```bash
# 克隆仓库
git clone https://github.com/suibuzhifanhua/lottery_manager.git
cd lottery_manager

# 安装依赖（使用阿里云镜像）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 运行
python main.py
```

## 📂 项目结构

```
lottery_manager/
├── main.py                 # 🖥️ 程序主入口（GUI 界面）
├── fetch_data.py           # 🌐 双色球数据抓取
├── fetch_dlt.py            # 🌐 大乐透数据抓取
├── predict.py              # 🤖 双色球预测模型
├── predict_dlt.py          # 🤖 大乐透预测模型
├── ledger.py               # 📒 购彩账本管理
├── gen_chart.py            # 📊 图表生成
├── feature_utils.py       # 🔧 特征工程工具
├── lottery_manager_optimized.spec  # 📦 PyInstaller 打包配置
├── requirements.txt       # 📦 依赖清单
└── dist/                  # 📦 打包输出目录
    └── lottery_manager_v2.4.0.exe  # 🎉 可执行文件
```

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| GUI | tkinter + tkcalendar |
| 数据处理 | pandas, numpy, scipy |
| 机器学习 | scikit-learn, xgboost, joblib |
| 深度学习 | PyTorch (CPU) |
| 可视化 | matplotlib |
| 打包 | PyInstaller |

## 📝 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v2.4.0 | 2026-04-30 | 🚀 启动速度优化，移除 OCR 模块，精简打包体积 |
| v2.3.3 | 2026-04-30 | 🔍 OCR 多注解析优化 |
| v2.3.0 | 2026-04-20 | ✨ 新增大乐透支持，新增 LSTM/RNN 预测模型 |

## ⚠️ 免责声明
第一次运行要加载模型分析数据，你也可以引入独显加速，反正我用9950x3d初始化大概3分钟左右。
本项目仅供**学习研究**使用。彩票开奖为随机事件，任何预测模型都无法保证中奖。请理性购彩，量力而行。

如果你用这个项目中了大奖，不妨请我喝杯咖啡。

<img width="422" height="208" alt="image" src="https://github.com/user-attachments/assets/04957376-48d7-4a83-a1a4-01ed31dde224" />


<img width="395" height="264" alt="image" src="https://github.com/user-attachments/assets/e41095e4-84ff-409a-be49-c053cb20a55b" />

