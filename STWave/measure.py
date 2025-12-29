import time
import torch
from memory_profiler import memory_usage
from datetime import datetime
import psutil

def measure_resources(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            mem_result = memory_usage((func, args, kwargs), max_usage=True, retval=True)
            if isinstance(mem_result, tuple) and len(mem_result) == 2:
                mem_usage_raw, result = mem_result
                if isinstance(mem_usage_raw, (list, tuple)):
                    mem_usage = max(mem_usage_raw) / 1024 if mem_usage_raw else 0
                else:
                    mem_usage = mem_usage_raw / 1024 if mem_usage_raw else 0
            else:
                print("⚠️ memory_usage returned unexpected format, using fallback")
                result = func(*args, **kwargs)
                mem_usage = 0
        except Exception as e:
            print(f"⚠️ memory_usage failed ({e}), using psutil fallback")
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 ** 3)
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / (1024 ** 3)
            mem_usage = max(mem_before, mem_after)

        end_time = time.time()
        execution_time = end_time - start_time

        # GPU 信息
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda:0")
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
            except Exception as e:
                print(f"⚠️ GPU memory monitoring failed: {e}")
                allocated = cached = 0
        else:
            allocated = cached = 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ===========================
        # 美观的表格式输出
        # ===========================
        report = f"""
============================================================
RESOURCE USAGE REPORT
============================================================
timestamp                     : {timestamp}
function_name                 : {func.__name__}
execution_time_minutes        : {execution_time/60:.4f}
execution_time_seconds        : {execution_time:.2f}
memory_usage_gb               : {mem_usage:.2f}
gpu_memory_allocated_gb       : {allocated:.2f}
gpu_memory_cached_gb          : {cached:.2f}
cuda_available                : {torch.cuda.is_available()}
============================================================
"""
        print(report)
        return result
    return wrapper
