import onnx
import numpy as np

from onnx_explorer import logo_str


def get_num_params(onnx_file_path):
    '''
    Get the number of parameters in an ONNX model
    :param onnx_file_path:
    :return:
    '''
    model = onnx.load(onnx_file_path)
    num_params = sum([np.prod(param.dims) for param in model.graph.initializer])
    return num_params


def estimate_memory_usage(num_params, param_dtype=np.float32, unit='MB'):
    '''
    Estimate the memory usage of an ONNX model
    :param num_params:
    :param param_dtype:
    :param unit:
    :return:
    '''
    bytes_per_param = np.dtype(param_dtype).itemsize
    memory_usage_bytes = num_params * bytes_per_param

    if unit == 'B':
        return memory_usage_bytes
    elif unit == 'KB':
        return memory_usage_bytes / 1024
    elif unit == 'MB':
        return memory_usage_bytes / (1024 * 1024)
    elif unit == 'GB':
        return memory_usage_bytes / (1024 * 1024 * 1024)
    else:
        raise ValueError("Invalid unit. Supported units are 'B', 'KB', and 'MB'.")


def print_memory_usage_bar(memory_usage, total_memory=4096):
    '''
    Print a progress bar for memory usage
    :param memory_usage: memory usage in MB
    :param total_memory: total memory in MB
    :return: None
    '''
    progress = int(memory_usage / total_memory * 100)
    progress_bar = f"[{'#' * progress}{'-' * (100 - progress)}] {progress}%"
    print(f"Memory usage: {progress_bar}")


def is_memory_overload(memory_usage, total_memory=4096):
    '''
    Check if memory usage exceeds total memory
    :param memory_usage: memory usage in MB
    :param total_memory: total memory in MB
    :return: True if memory usage is greater than total memory, False otherwise
    '''
    return memory_usage > total_memory


if __name__ == '__main__':
    print(f"{logo_str}\n")
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    model_name = input_model_path.split('/')[-1].split('.')[0]
    num_params = get_num_params(input_model_path)
    total_memory = 4096
    data_types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]

    print(f"Number of parameters: {num_params}")
    print("Estimated memory usage for different data types:")

    for dtype in data_types:
        memory_usage_mb = estimate_memory_usage(num_params, param_dtype=dtype, unit='MB')
        print(f"【{model_name}】-> {dtype}: {memory_usage_mb:.2f} MB")
        print('-' * (120))
        print_memory_usage_bar(memory_usage_mb, total_memory)

        if is_memory_overload(memory_usage_mb):
            print("Warning: Memory overload!")