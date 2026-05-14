# -*- mode: python ; coding: utf-8 -*-
"""
彩票预测系统打包配置
支持双色球和大乐透预测，包含所有优化模块

⚠️ 打包步骤：
1. 修改下方的 VERSION 和 BUILD_DATE 为当前版本
2. 运行: python -m PyInstaller lottery_manager_optimized.spec --clean
3. 生成的 exe 在 dist/ 目录，文件名自动包含版本号
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_submodules

block_cipher = None

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(SPEC))

# ─── XGBoost DLL 路径 ─────────────────────────────────────────
# 从 Python 环境查找 xgboost.dll
xgboost_lib_dir = None
for p in sys.path:
    lib_path = os.path.join(p, "xgboost", "lib")
    if os.path.exists(lib_path):
        xgboost_lib_dir = lib_path
        break

# ─── 收集 XGBoost 所有数据文件 ─────────────────────────────────
xgboost_datas = collect_data_files("xgboost")

a = Analysis(
    ['main.py'],
    pathex=[current_dir],
    binaries=[
        # XGBoost 核心 DLL（从 xgboost/lib 目录复制）
        (os.path.join(xgboost_lib_dir, 'xgboost.dll'), '.'),
    ] if xgboost_lib_dir and os.path.exists(os.path.join(xgboost_lib_dir, 'xgboost.dll')) else [],
    datas=xgboost_datas + [
        # 包含 models 目录（如果有预训练模型）
        # ('models', 'models'),
    ],
    hiddenimports=[
        # 核心模块
        'fetch_data',
        'fetch_dlt',
        'feature_utils',
        'predict',
        'predict_dlt',
        'ledger',
        'gen_chart',
        # 机器学习模块
        'sklearn',
        'sklearn.ensemble',
        'sklearn.neural_network',
        'sklearn.preprocessing',
        'sklearn.utils',
        'sklearn.utils._openmp_helpers',
        'xgboost',
        'xgboost.core',
        'joblib',
        'joblib.externals.cloudpickle',
        'joblib.externals.loky',
        'scipy',
        'scipy.sparse',
        'scipy.spatial',
        # 图形和UI
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.backends',
        'matplotlib.backends.tkagg',
        'PIL',
        'PIL._imaging',
        # 数据处理
        'numpy',
        'numpy.core',
        'numpy.random',
        'pandas',
        'openpyxl',
        # 工具库
        'requests',
        'sqlite3',
        'collections',
        'concurrent.futures',
        # 日期处理
        'datetime',
        # 可能的 tkcalendar
        'tkcalendar',
        # 其他可能的依赖
        'dateutil',
        'dateutil.parser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        'disable_torch_dynamo.py',  # 禁用 PyTorch dynamo 编译（打包后用户名缺失）
    ],
    excludes=[
        'pytest',
        'IPython',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ─── 版本号配置 ────────────────────────────────────────────────
# ⚠️ 每次打包前请更新此版本号！
VERSION = "2.4.0"
BUILD_DATE = "2026-04-30"

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=f'lottery_manager_v{VERSION}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'] if os.path.exists('icon.ico') else None,
)
