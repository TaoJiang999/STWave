import time
import torch
from memory_profiler import memory_usage
import os
import csv
from datetime import datetime


def measure_resources(func):
    """修复版本的资源监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # 方法1: 使用 retval=True 来获取函数返回值，避免重复执行
        try:
            # memory_usage 返回 (内存使用值, 函数返回值) 当 retval=True
            mem_result = memory_usage((func, args, kwargs), max_usage=True, retval=True)
            
            # 检查返回值的类型和结构
            print(f"Debug: memory_usage returned: {type(mem_result)}, value: {mem_result}")
            
            if isinstance(mem_result, tuple) and len(mem_result) == 2:
                # 正常情况：(内存使用, 函数返回值)
                mem_usage_raw, result = mem_result
                if isinstance(mem_usage_raw, (list, tuple)):
                    mem_usage = max(mem_usage_raw) / 1024 if mem_usage_raw else 0
                else:
                    mem_usage = mem_usage_raw / 1024 if mem_usage_raw else 0
            else:
                # 异常情况：降级到简单监控
                print("produre is stop")
                # result = func(*args, **kwargs)
                # mem_usage = 0  # 无法准确测量内存
                
        except Exception as e:
            print(f"Warning: memory_usage failed ({e}), using simple monitoring")
            # 降级方案：使用psutil监控
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 ** 3)  # GB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / (1024 ** 3)  # GB
            mem_usage = max(memory_before, memory_after)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # GPU内存监控
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda:0")  # 默认使用GPU 0
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
            except Exception as e:
                print(f"Warning: GPU memory monitoring failed: {e}")
                allocated = cached = 0
        else:
            allocated = cached = 0
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 打印信息（保持原有格式）
        print(f"Function '{func.__name__}' executed in {execution_time/60:.4f} minutes.")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")
        
        # 准备要保存的数据
        resource_data = {
            'timestamp': timestamp,
            'function_name': func.__name__,
            'execution_time_minutes': round(execution_time/60, 4),
            'execution_time_seconds': round(execution_time, 2),
            'memory_usage_gb': round(mem_usage, 2),
            'gpu_memory_allocated_gb': round(allocated, 2),
            'gpu_memory_cached_gb': round(cached, 2),
            'cuda_available': torch.cuda.is_available(),
            'device_idx': 0  # 默认GPU 0
        }
        
        # 保存到文件
        save_resource_data(resource_data)
        
        return result
    return wrapper

def save_resource_data(data):
    """
    将资源使用数据保存到文件中
    支持CSV和JSON两种格式
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义文件路径
    csv_file = os.path.join(current_dir, 'resource_usage.csv')
    # json_file = os.path.join(current_dir, 'resource_usage.json')
    
    # 保存为CSV格式
    save_to_csv(csv_file, data)
    
    # 保存为JSON格式

def save_to_csv(file_path, data):
    """保存数据到CSV文件"""
    # CSV文件的列头
    fieldnames = [
        'timestamp', 'function_name', 'execution_time_minutes', 
        'execution_time_seconds', 'memory_usage_gb', 
        'gpu_memory_allocated_gb', 'gpu_memory_cached_gb', 
        'cuda_available', 'device_idx'
    ]
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
                print(f"Created new CSV file: {file_path}")
            
            # 写入数据
            writer.writerow(data)
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")



# 使用示例
if __name__ == "__main__":
    # 模拟device_idx变量（在你的实际代码中这个变量应该已经定义）
    device_idx = 0
    
    # 示例函数1
    @measure_resources
    def example_function1():
        import time
        time.sleep(2)  # 模拟耗时操作
        return "Function 1 completed"
    
    # 示例函数2  
    @measure_resources
    def example_function2():
        import numpy as np
        # 模拟内存密集型操作
        data = np.random.randn(1000, 1000)
        result = np.dot(data, data.T)
        return result
    
    # 执行示例
    print("执行示例函数...")
    result1 = example_function1()
    result2 = example_function2()
    

# @measure_resources
# def preprocess():
#     for i in range(100000):
#         a = 5
#         b = 10
#         a = a+b

# preprocess()