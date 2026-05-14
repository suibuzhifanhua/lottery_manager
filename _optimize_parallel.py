"""
辅助脚本：优化 auto_predict.py 的并行预测
"""
import re

# 读取原文件
with open(r'C:\Users\Fisheep\Desktop\成品\cp\lottery_manager\auto_predict.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 添加并行预测辅助函数
parallel_helper = '''

def run_parallel_predict(db_path, methods, module, max_workers=4):
    """
    🚀 优化版：并行执行多个预测方案

    Args:
        db_path: 数据库路径
        methods: [(method_key, method_name), ...]
        module: predict 或 predict_dlt 模块
        max_workers: 最大并行数

    Returns:
        dict: {method_key: result, ...}
    """
    results = {}

    def predict_one(method_key, label):
        try:
            result = module.predict_next_issue(db_path, method=method_key)
            return method_key, label, result, None
        except Exception as e:
            return method_key, label, None, str(e)

    print(f"  [Parallel] Starting {len(methods)} predictions with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(predict_one, method_key, label): (method_key, label)
            for method_key, label in methods
        }

        completed = 0
        for future in as_completed(futures):
            method_key, label, result, error = future.result()
            completed += 1

            if error:
                print(f"  [{completed}/{len(methods)}] {label}: [FAIL] {error[:40]}")
                results[method_key] = {"error": error}
            elif result and "error" not in result:
                print(f"  [{completed}/{len(methods)}] {label}: [OK]")
                results[method_key] = result
            else:
                err = result.get("error", "Unknown") if result else "Empty"
                print(f"  [{completed}/{len(methods)}] {label}: [FAIL] {err[:40]}")
                results[method_key] = result or {"error": err}

    return results


'''

# 在 run_with_progress 函数后插入并行函数
content = content.replace(
    'def build_report_ssq',
    parallel_helper + '\ndef build_report_ssq'
)

# 修改 main 函数中的预测执行部分
# 找到并替换双色球预测部分
old_ssq_predict = '''    # ── 3. 双色球预测 ──────────────────────────────────────
    print("【3/5】正在执行双色球预测（共12种方案）...")
    print("-" * W)
    results_ssq = {}
    for method, label in _METHODS:
        results_ssq[method] = run_with_progress(label, DB_PATH, method, predict)
    print()'''

new_ssq_predict = '''    # ── 3. 双色球预测（🚀 并行优化）───────────────────────
    print("【3/5】正在执行双色球预测（共12种方案）...")
    print("-" * W)
    results_ssq = run_parallel_predict(DB_PATH, _METHODS, predict, max_workers=4)
    print()'''

content = content.replace(old_ssq_predict, new_ssq_predict)

# 找到并替换大乐透预测部分
old_dlt_predict = '''    # ── 4. 大乐透预测 ──────────────────────────────────────
    print("【4/5】正在执行大乐透预测（共12种方案）...")
    print("-" * W)
    results_dlt = {}
    for method, label in _METHODS:
        results_dlt[method] = run_with_progress(label, DB_PATH, method, predict_dlt)
    print()'''

new_dlt_predict = '''    # ── 4. 大乐透预测（🚀 并行优化）───────────────────────
    print("【4/5】正在执行大乐透预测（共12种方案）...")
    print("-" * W)
    results_dlt = run_parallel_predict(DB_PATH, _METHODS, predict_dlt, max_workers=4)
    print()'''

content = content.replace(old_dlt_predict, new_dlt_predict)

# 写回文件
with open(r'C:\Users\Fisheep\Desktop\成品\cp\lottery_manager\auto_predict.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] auto_predict.py parallel optimization done!")
