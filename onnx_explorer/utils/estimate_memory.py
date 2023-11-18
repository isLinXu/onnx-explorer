import onnx
import numpy as np

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


if __name__ == '__main__':
    # input_model_path = "/path/to/your/input/onnx_model.onnx"
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    num_params = get_num_params(input_model_path)

    data_types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]

    print(f"Number of parameters: {num_params}")
    print("Estimated memory usage for different data types:")

    for dtype in data_types:
        memory_usage_mb = estimate_memory_usage(num_params, param_dtype=dtype, unit='MB')
        print(f"{dtype}: {memory_usage_mb:.2f} MB")
