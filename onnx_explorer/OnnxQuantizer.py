import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

class ONNXModelQuantizer:

    @staticmethod
    def quantize_onnx_model(input_model_path, output_model_path, quant_type=QuantType.QInt8):
        # Load the ONNX model
        model = onnx.load(input_model_path)

        # Quantize the ONNX model
        quantized_model = quantize_dynamic(model_input=model, model_output=output_model_path, op_types_to_quantize=quant_type)

        # Save the quantized ONNX model
        onnx.save(quantized_model, output_model_path)

if __name__ == '__main__':
    # input_model_path = "/path/to/your/input/onnx_model.onnx"
    # output_model_path = "/path/to/your/output/quantized_onnx_model.onnx"
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_model_path = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_quantizer.onnx"
    ONNXModelQuantizer.quantize_onnx_model(input_model_path, output_model_path)