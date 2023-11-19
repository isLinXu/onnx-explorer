
import onnx
import numpy as np
import torch

from onnx_explorer import logo_str


class ModelEstimate:
    def __init__(self, model_file_path=None, model_type='onnx', manual_num_params=None, total_memory=4, total_memory_unit='GB'):
        '''
        Initialize the ModelEstimate class
        :param model_file_path:
        :param model_type:
        :param manual_num_params:
        '''
        if manual_num_params is None and model_file_path is None:
            raise ValueError("Either model_file_path or manual_num_params must be provided.")
        elif manual_num_params is not None and model_file_path is not None:
            raise ValueError("Only one of model_file_path and manual_num_params must be provided.")
        elif manual_num_params is not None:
            print("Warning: Using manual_num_params. The estimated memory usage may not be accurate.")
            self.model_name = "Manual"
        elif model_file_path is not None:
            self.model_name = model_file_path.split('/')[-1].split('.')[0] if model_file_path else "Manual"
        self.total_memory = total_memory
        self.total_memory_unit = total_memory_unit
        self.print_model_info()
        if manual_num_params is not None:
            self.num_params = manual_num_params
        else:
            self.model_type = model_type
            if model_type == 'onnx':
                self.model = onnx.load(model_file_path)
                self.num_params = self.get_num_params_onnx()
                self.layers = self.get_model_layers()
            elif model_type == 'pt':
                self.model = torch.load(model_file_path)
                self.num_params = self.get_num_params_pt()
            else:
                raise ValueError("Invalid model_type. Supported types are 'onnx' and 'pt'.")

    def get_num_params_onnx(self):
        '''
        Get the number of parameters in the ONNX model
        :return:
        '''
        num_params = sum([np.prod(param.dims) for param in self.model.graph.initializer])
        print(f"Number of parameters: {num_params}")
        return num_params

    def get_num_params_pt(self):
        '''
        Get the number of parameters in the PyTorch model
        :return:
        '''
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")
        return num_params

    def print_model_info(self):
        print(f"{logo_str}\n")
        print(f"【{self.model_name}】")

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
        total_memory_bytes = ModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
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
        total_memory_bytes = ModelEstimate.convert_memory(total_memory, total_memory_unit, 'B')
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
        if type(value) == str:
            value = float(value)
        return value * conversion_factors[input_unit] / conversion_factors[output_unit]


    def get_estimated_memory_usage(self):
        '''
        Get the estimated memory usage for different data types
        :return:
        '''
        print("Estimated memory usage for different data types:")
        data_types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
        for dtype in data_types:
            memory_usage_mb = self.estimate_memory_usage(param_dtype=dtype, unit=self.total_memory_unit)
            memory_usage = self.convert_memory(memory_usage_mb, self.total_memory_unit, 'B')
            print(f"【{self.model_name}】-> {dtype.__name__}: -> {memory_usage_mb:.2f} {self.total_memory_unit} in {self.total_memory}{self.total_memory_unit}")
            print('-' * (120))
            ModelEstimate.print_memory_usage_bar(memory_usage, self.total_memory, self.total_memory_unit)

            if ModelEstimate.is_memory_overload(memory_usage, self.total_memory, self.total_memory_unit):
                print("Warning: Memory overload!")


def main():
    # init params
    input_model_path_onnx = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov5/yolov5x6.onnx"
    input_model_path_pt = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov5/yolov5x6.pt"
    model_name = input_model_path_onnx.split('/')[-1].split('.')[0]
    # model_name = 'my_model'
    total_memory = 4
    total_memory_unit = 'GB'

    model = ModelEstimate(input_model_path_onnx, model_type='onnx', total_memory=total_memory, total_memory_unit=total_memory_unit)
    model.get_estimated_memory_usage()

    # model = ModelEstimate(input_model_path_pt, model_type='pt')
    # model.get_estimated_memory_usage()

    # Example with manual input of parameters
    # manual_params = 500000000  # 50 million parameters
    # model = ModelEstimate(manual_num_params=manual_params)
    # model.get_estimated_memory_usage()


if __name__ == '__main__':
    main()