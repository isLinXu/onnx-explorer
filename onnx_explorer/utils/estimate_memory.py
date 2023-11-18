
import onnx
import numpy as np

class ONNXModelEstimate:
    def __init__(self, onnx_file_path):
        self.model = onnx.load(onnx_file_path)
        self.num_params = self.get_num_params()

    def get_num_params(self):
        return sum([np.prod(param.dims) for param in self.model.graph.initializer])

    def estimate_memory_usage(self, param_dtype=np.float32, unit='MB'):
        bytes_per_param = np.dtype(param_dtype).itemsize
        memory_usage_bytes = self.num_params * bytes_per_param

        if unit == 'B':
            return memory_usage_bytes
        elif unit == 'KB':
            return memory_usage_bytes / 1024
        elif unit == 'MB':
            return memory_usage_bytes / (1024 * 1024)
        elif unit == 'GB':
            return memory_usage_bytes / (1024 * 1024 * 1024)
        else:
            raise ValueError("Invalid unit. Supported units are 'B', 'KB', 'MB', and 'GB'.")

    @staticmethod
    def print_memory_usage_bar(memory_usage, total_memory, total_memory_unit='MB', bar_length=100):
        total_memory_bytes = ONNXModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
        progress = int(memory_usage / total_memory_bytes * bar_length)
        progress_bar = f"[{'#' * progress}{'-' * (bar_length - progress)}] {progress}%"
        print(f"Memory usage: {progress_bar}")

    @staticmethod
    def is_memory_overload(memory_usage, total_memory, total_memory_unit='MB'):
        total_memory_bytes = ONNXModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
        return memory_usage > total_memory_bytes

    @staticmethod
    def convert_memory(value, input_unit, output_unit):
        conversion_factors = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        return value * conversion_factors[input_unit] / conversion_factors[output_unit]

if __name__ == '__main__':
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    model_name = input_model_path.split('/')[-1].split('.')[0]
    onnx_model = ONNXModelEstimate(input_model_path)
    total_memory = 4
    total_memory_unit = 'GB'
    data_types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]

    print(f"Number of parameters: {onnx_model.num_params}")
    print("Estimated memory usage for different data types:")

    for dtype in data_types:
        memory_usage_mb = onnx_model.estimate_memory_usage(param_dtype=dtype, unit='MB')
        memory_usage = onnx_model.convert_memory(memory_usage_mb, 'MB', 'B')
        print(f"【{model_name}】-> {dtype}: {memory_usage_mb:.2f} MB")
        print('-' * (120))
        ONNXModelEstimate.print_memory_usage_bar(memory_usage, total_memory, total_memory_unit)

        if ONNXModelEstimate.is_memory_overload(memory_usage, total_memory, total_memory_unit):
            print("Warning: Memory overload!")