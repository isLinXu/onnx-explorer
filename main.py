from onnx_explorer.OnnxAlyzer import ONNXModelAnalyzer
from onnx_explorer import logo_str

if __name__ == '__main__':
    print(f"{logo_str}\n")
    # 使用示例
    # Usage example
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_file = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_analysis"
    ONNXModelAnalyzer.analyze_onnx_model(model_path, save_to_file=True, output_file=output_file, show_node_details=False)