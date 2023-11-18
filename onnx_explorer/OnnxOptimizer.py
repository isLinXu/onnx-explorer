
import onnx
from onnxoptimizer import optimize
import onnxsim

from onnx_explorer.utils import get_file_size

class ONNXModelOptimizer:

    @staticmethod
    def optimize_onnx_model(input_model_path, output_model_path, use_onnxoptimizer=True, use_onnxsim=True):
        # Load the ONNX model
        model = onnx.load(input_model_path)
        model_path_size = get_file_size(input_model_path)
        print("after model_size:", model_path_size)

        # Optimize the ONNX model using onnxoptimizer
        if use_onnxoptimizer:
            passes = ['extract_constant_to_initializer',
                      'eliminate_unused_initializer',
                      'eliminate_identity']
            model = optimize(model, passes)
            print("Optimized with onnxoptimizer")

        # Optimize the ONNX model using onnx-simplifier
        if use_onnxsim:
            model, check = onnxsim.simplify(model)
            if check:
                print("Optimized with onnx-simplifier")
            else:
                print("Failed to optimize with onnx-simplifier")

        # Save the optimized ONNX model
        onnx.save(model, output_model_path)
        output_model_path_size = get_file_size(output_model_path)
        print("after model_size:", output_model_path_size)

if __name__ == '__main__':
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_model_path = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_optimized.onnx"
    ONNXModelOptimizer.optimize_onnx_model(input_model_path, output_model_path,use_onnxoptimizer=True, use_onnxsim=True)
