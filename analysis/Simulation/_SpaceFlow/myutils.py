import time
import torch
from memory_profiler import memory_usage
import os
import csv
from datetime import datetime


def measure_resources(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        
        try:
           
            mem_result = memory_usage((func, args, kwargs), max_usage=True, retval=True)
            
            
            print(f"Debug: memory_usage returned: {type(mem_result)}, value: {mem_result}")
            
            if isinstance(mem_result, tuple) and len(mem_result) == 2:
                
                mem_usage_raw, result = mem_result
                if isinstance(mem_usage_raw, (list, tuple)):
                    mem_usage = max(mem_usage_raw) / 1024 if mem_usage_raw else 0
                else:
                    mem_usage = mem_usage_raw / 1024 if mem_usage_raw else 0
            else:
                
                print("Warning: memory_usage returned unexpected format, using simple monitoring")
                
                
        except Exception as e:
            print(f"Warning: memory_usage failed ({e}), using simple monitoring")
           
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 ** 3)  # GB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / (1024 ** 3)  # GB
            mem_usage = max(memory_before, memory_after)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        
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
        
       
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        print(f"Function '{func.__name__}' executed in {execution_time/60:.4f} minutes.")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")
        
        
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
        
     
        save_resource_data(resource_data)
        
        return result
    return wrapper

def save_resource_data(data):
 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    csv_file = os.path.join(current_dir, 'resource_usage.csv')
    # json_file = os.path.join(current_dir, 'resource_usage.json')
    
  
    save_to_csv(csv_file, data)
    
   

def save_to_csv(file_path, data):


    fieldnames = [
        'timestamp', 'function_name', 'execution_time_minutes', 
        'execution_time_seconds', 'memory_usage_gb', 
        'gpu_memory_allocated_gb', 'gpu_memory_cached_gb', 
        'cuda_available', 'device_idx'
    ]
    

    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            

            if not file_exists:
                writer.writeheader()
                print(f"Created new CSV file: {file_path}")
            

            writer.writerow(data)
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")




if __name__ == "__main__":

    device_idx = 0
    

    @measure_resources
    def example_function1():
        import time
        time.sleep(2)  
        return "Function 1 completed"
    
  
    @measure_resources
    def example_function2():
        import numpy as np

        data = np.random.randn(1000, 1000)
        result = np.dot(data, data.T)
        return result
    

    result1 = example_function1()
    result2 = example_function2()
    
