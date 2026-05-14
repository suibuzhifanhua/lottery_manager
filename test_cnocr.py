"""测试 cnocr 是否正常工作"""
import os
import sys

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("正在测试 cnocr...")

try:
    from cnocr import CnOcr
    print("CnOcr 导入成功，正在初始化...")
    ocr = CnOcr()
    print("CnOcr 初始化成功！")
except Exception as e:
    print(f"CnOcr 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("测试完成！")
