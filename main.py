import argparse

from onnx_explorer.OnnxAlyzer import ONNXModelAnalyzer
from onnx_explorer import logo_str


if __name__ == '__main__':
    print(f"{logo_str}\n")
    # 使用示例
    '''
    # Usage example
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_file = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_analysis"
    ONNXModelAnalyzer.analyze_onnx_model(model_path, save_to_file=True, output_file=output_file, show_node_details=False)
    '''
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Analyze ONNX model.")
    parser.add_argument("-m","--model-path", help="Path to the ONNX model file.")
    parser.add_argument("-s", "--save", action="store_true", help="Save analysis to a text file.")
    parser.add_argument("-o", "--output-path", help="Path to the output text file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show node details.")
    args = parser.parse_args()
    model_path = args.model_path
    save_to_file = args.save
    output_file = args.output_path
    show_node_details = args.verbose
    ONNXModelAnalyzer.analyze_onnx_model(model_path, save_to_file=save_to_file, output_file=output_file, show_node_details=show_node_details)
