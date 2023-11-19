import argparse

from onnx_explorer.utils import estimate_memory
from onnx_explorer.utils.estimate_memory import ModelEstimate

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Analyze ONNX model.")
    parser.add_argument("-m", "--model-path", help="Path to the ONNX model file.")
    parser.add_argument("-t", "--model-type", default='onnx', help="Type of the model. Supported types are 'onnx' and 'pt'.")
    parser.add_argument("-n", "--num_memory", default=4, help="Total memory size.")
    parser.add_argument("-u", "--unit_memory", default='GB', help="Total memory unit.")
    parser.add_argument("-p", "--manual_params", default=500000000, type=int, help="Number of parameters in the model.")
    args = parser.parse_args()
    model_path = args.model_path
    model_type = args.model_type
    num_memory = args.num_memory
    unit_memory = args.unit_memory
    manual_params = args.manual_params

    if unit_memory not in ['B', 'KB', 'MB', 'GB']:
        raise ValueError("Invalid memory unit. Supported units are 'B', 'KB', 'MB', 'GB'.")

    if model_path is None:
        print("manual_params:", manual_params)
        model = ModelEstimate(total_memory=num_memory,
                              total_memory_unit=unit_memory,
                              manual_num_params=manual_params)
        model.get_estimated_memory_usage()
    else:
        model = ModelEstimate(model_file_path=model_path,
                              model_type=model_type,
                              total_memory=num_memory,
                              total_memory_unit=unit_memory)
        model.get_estimated_memory_usage()