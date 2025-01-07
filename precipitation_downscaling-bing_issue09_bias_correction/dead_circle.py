import threading
import time
import torch
import os
import psutil

# 检查GPU使用情况
def is_gpu_busy(threshold=30):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认使用第一个GPU
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        # print(f"当前GPU使用率: {gpu_util}%")
        return gpu_util > threshold
    except ModuleNotFoundError:
        # print("pynvml 未安装，跳过GPU利用率检测。")
        return False
    except Exception as e:
        # print(f"GPU检测异常: {str(e)}")
        return False

# 检查CPU使用情况
def is_cpu_busy(threshold=50):
    cpu_usage = psutil.cpu_percent(interval=1)
    # print(f"当前CPU使用率: {cpu_usage}%")
    return cpu_usage > threshold

# 检查内存使用情况
def is_memory_busy(threshold=70):
    memory = psutil.virtual_memory()
    # print(f"当前内存使用率: {memory.percent}%")
    return memory.percent > threshold

# 占用 CPU 线程
def keep_cpu_busy():
    while True:
        if not is_cpu_busy():
            pass
        else:
            time.sleep(10)

# 占用内存
def allocate_memory(size_mb):
    data = None
    while True:
        if not is_memory_busy():
            data = bytearray(size_mb * 1024 * 1024)
        else:
            data = None
        time.sleep(10)

# 增强版 GPU 占用函数
def keep_gpu_busy(device="cuda:0"):
    if torch.cuda.is_available():
        matrices = [torch.randn(20000, 20000, device=device) for _ in range(3)]  # 创建3个大矩阵，每个约3.2GB显存
        # print(f"在 {device} 上分配了3个20000x20000的矩阵，占用约{len(matrices)*3.2:.1f}GB显存")

        while True:
            if not is_gpu_busy():
                for _ in range(5):  # 增加运算次数，每次进行5次矩阵乘法
                    matrices = [x @ x for x in matrices]  
                    torch.cuda.synchronize()
            else:
                # print("GPU忙碌，暂停2秒")
                time.sleep(2)
    else:
        print("未检测到GPU，跳过GPU占用部分。")

# 启动 CPU 和内存占用
num_cpu_threads = 5
memory_to_allocate = 8192  # 占用8GB内存

for i in range(num_cpu_threads):
    thread = threading.Thread(target=keep_cpu_busy)
    thread.start()

thread = threading.Thread(target=allocate_memory, args=(memory_to_allocate,))
thread.start()

# 启动 GPU 占用线程
thread = threading.Thread(target=keep_gpu_busy)
thread.start()
