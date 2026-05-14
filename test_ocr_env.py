"""测试打包后的 OCR 功能"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 检查能否导入 cnocr
try:
    from cnocr import CnOcr
    print("[OK] cnocr import success")
except ImportError as e:
    print(f"[FAIL] cnocr import failed: {e}")
    sys.exit(1)

# 检查模型文件
model_paths = [
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "cnstd", "1.2", "ppocr"),
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "cnocr", "2.3"),
]

for p in model_paths:
    if os.path.exists(p):
        print(f"[OK] Model dir exists: {p}")
    else:
        print(f"[FAIL] Model dir not found: {p}")

# 尝试初始化 CnOcr
try:
    print("Initializing CnOcr...")
    ocr = CnOcr()
    print("[OK] CnOcr init success")
except Exception as e:
    print(f"[FAIL] CnOcr init failed: {e}")
    import traceback
    traceback.print_exc()

print("Test complete!")
