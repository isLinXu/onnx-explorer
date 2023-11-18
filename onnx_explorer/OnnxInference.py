
import onnxruntime
import numpy as np

class ONNXModelInference:

    @staticmethod
    def onnx_dtype_to_numpy_dtype(onnx_dtype):
        '''
        Convert ONNX data type to Numpy data type
        :param onnx_dtype:
        :return:
        '''
        if onnx_dtype == 'tensor(float)':
            return np.float32
        elif onnx_dtype == 'tensor(double)':
            return np.float64
        elif onnx_dtype == 'tensor(int32)':
            return np.int32
        elif onnx_dtype == 'tensor(int64)':
            return np.int64
        else:
            raise ValueError(f"Unsupported ONNX data type: {onnx_dtype}")

    @staticmethod
    def prepare_input(input_data, input_name, input_shape, input_dtype):
        '''
        Prepare input data for ONNX model inference
        :param input_data:
        :param input_name:
        :param input_shape:
        :param input_dtype:
        :return:
        '''
        # Convert ONNX data type to Numpy data type
        input_dtype_np = ONNXModelInference.onnx_dtype_to_numpy_dtype(input_dtype)

        # Reshape the input data if necessary
        if input_data.shape != tuple(input_shape):
            input_data = np.reshape(input_data, input_shape)

        # Convert the input data to the required data type
        input_data = input_data.astype(input_dtype_np)

        return {input_name: input_data}

    @staticmethod
    def run_inference(input_model_path, input_data):
        '''
        Run inference on an ONNX model
        :param input_model_path:
        :param input_data:
        :return:
        '''
        # Create an ONNX runtime session
        session = onnxruntime.InferenceSession(input_model_path)

        # Get input and output names, shapes, and data types
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        input_name = input_info.name
        output_name = output_info.name

        input_shape = input_info.shape
        input_dtype = input_info.type

        # Prepare input data
        input_data_prepared = ONNXModelInference.prepare_input(input_data, input_name, input_shape, input_dtype)

        # Run inference
        result = session.run([output_name], input_data_prepared)

        return result

if __name__ == '__main__':
    input_model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"

    # Create dummy input data
    # input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Run inference
    result = ONNXModelInference.run_inference(input_model_path, input_data)
    print("Inference result:", result)