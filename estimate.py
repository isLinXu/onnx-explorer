import argparse

from onnx_explorer.utils import estimate_memory
from onnx_explorer.utils.estimate_memory import ModelEstimate

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Analyze ONNX model.")
    parser.add_argument("-m", "--model-path", help="Path to the ONNX model file.")
    parser.add_argument("-n", "--num_memory", default=4, help="Total memory size.")
    parser.add_argument("-u", "--unit_memory", default='GB', help="Total memory unit.")
    args = parser.parse_args()
    model_path = args.model_path
    num_memory = args.num_memory
    unit_memory = args.unit_memory
    model = ModelEstimate(model_path, model_type='onnx',
                          total_memory=num_memory,
                          total_memory_unit=unit_memory)
    model.get_estimated_memory_usage()
