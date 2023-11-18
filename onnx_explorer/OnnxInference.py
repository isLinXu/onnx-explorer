import onnxruntime
import numpy as np

class ONNXModelInference:

    @staticmethod
    def run_inference(input_model_path, input_data):
        # Create an ONNX runtime session
        session = onnxruntime.InferenceSession(input_model_path)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        result = session.run([output_name], {input_name: input_data})

        return result

if __name__ == '__main__':
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"

    # Create dummy input data
    # input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Run inference
    result = ONNXModelInference.run_inference(input_model_path, input_data)
    print("Inference result:", result)