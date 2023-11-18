
import onnx
import numpy as np

from onnx_explorer import logo_str


class ONNXModelEstimate:
    def __init__(self, onnx_file_path):
        self.print_model_info()
        self.model = onnx.load(onnx_file_path)
        self.num_params = self.get_num_params()
        self.layers = self.get_model_layers()
        self.model_name = onnx_file_path.split('/')[-1].split('.')[0]

    def print_model_info(self):
        print(f"{logo_str}\n")
        print(f"【{model_name}】")

    def get_model_layers(self):
        '''
        Get the layers of the ONNX model
        :return:
        '''
        layers_len = len(self.model.graph.node)
        print("Layers:", layers_len)
        return layers_len

    def get_num_params(self):
        '''
        Get the number of parameters in the ONNX model
        :return:
        '''
        num_params = sum([np.prod(param.dims) for param in self.model.graph.initializer])
        print(f"Number of parameters: {num_params}")
        return num_params

    def estimate_memory_usage(self, param_dtype=np.float32, unit='MB'):
        '''
        Estimate the memory usage of the ONNX model
        :param param_dtype:
        :param unit:
        :return:
        '''
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
        '''
        Print a progress bar for memory usage
        :param memory_usage:
        :param total_memory:
        :param total_memory_unit:
        :param bar_length:
        :return:
        '''
        total_memory_bytes = ONNXModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
        progress = int(memory_usage / total_memory_bytes * bar_length)
        progress_bar = f"[{'#' * progress}{'-' * (bar_length - progress)}] {progress}%"
        print(f"Memory usage: {progress_bar}")

    @staticmethod
    def is_memory_overload(memory_usage, total_memory, total_memory_unit='MB'):
        '''
        Check if the memory usage exceeds the total memory
        :param memory_usage:
        :param total_memory:
        :param total_memory_unit:
        :return:
        '''
        total_memory_bytes = ONNXModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
        return memory_usage > total_memory_bytes

    @staticmethod
    def convert_memory(value, input_unit, output_unit):
        '''
        Convert memory from one unit to another
        :param value:
        :param input_unit:
        :param output_unit:
        :return:
        '''
        conversion_factors = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        return value * conversion_factors[input_unit] / conversion_factors[output_unit]

    @staticmethod
    def get_estimated_memory_usage():
        '''
        Get the estimated memory usage for different data types
        :return:
        '''
        print("Estimated memory usage for different data types:")
        for dtype in data_types:
            memory_usage_mb = onnx_model.estimate_memory_usage(param_dtype=dtype, unit='MB')
            memory_usage = onnx_model.convert_memory(memory_usage_mb, 'MB', 'B')
            print(f"【{model_name}】-> {dtype.__name__}: -> {memory_usage_mb:.2f} MB in {total_memory}{total_memory_unit}")
            print('-' * (120))
            ONNXModelEstimate.print_memory_usage_bar(memory_usage, total_memory, total_memory_unit)

            if ONNXModelEstimate.is_memory_overload(memory_usage, total_memory, total_memory_unit):
                print("Warning: Memory overload!")


if __name__ == '__main__':
    # init params
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    model_name = input_model_path.split('/')[-1].split('.')[0]
    total_memory = 4
    total_memory_unit = 'GB'
    data_types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]

    onnx_model = ONNXModelEstimate(input_model_path)
    onnx_model.get_estimated_memory_usage()