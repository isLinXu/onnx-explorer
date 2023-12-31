import onnx
import numpy as np
import onnx.helper as helper

class OnnxTypeModifier:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)

    def update_node_data_type(self, update_list, new_data_type):
        new_data_type_number = self.convert_data_type_to_number(new_data_type)
        self.model = self.update_onnx_node_data_type(self.model, update_list, new_data_type_number)

    def save_model(self, output_path):
        onnx.save(self.model, output_path)

    def update_onnx_node_data_type(self, model, update_list, new_data_type):
        graph = model.graph
        inputs = graph.input

        for input in model.graph.node:
            input_name = input.name
            output_name = input.output[0]

            if input_name in update_list:
                input.attribute[0].t.data_type = new_data_type
        return model

    def convert_data_type_to_number(self, data_type_str):
        data_type_map = {
            "float16": 10,
            "float": 1,  # float32
            "float32": 1,
            "float64": 11,
            "int8": 3,
            "int16": 5,
            "int": 6,  # int32
            "int32": 6,
            "int64": 7,
            "uint8": 2,
            "uint16": 4,
            "complex64": 12,
            "complex128": 13,
            "string": 8,
            "bool": 9,
        }
        return data_type_map[data_type_str]

if __name__ == "__main__":
    model_path = "/Users/gatilin/youtu-work/SVAP/modified_det_model_float32.onnx"
    update_list = [
        '/detect/Constant_8', '/detect/Constant_9', '/detect/Constant_10', '/detect/Constant_11', '/detect/Constant_12',
        '/detect/Constant_14',
        '/detect/Constant_15', '/detect/Constant_16'
    ]
    new_data_type = "float32"  # 设置新的数据类型
    output_path = "output.onnx"

    onnx_modifier = OnnxTypeModifier(model_path)
    onnx_modifier.update_node_data_type(update_list, new_data_type)
    onnx_modifier.save_model(output_path)