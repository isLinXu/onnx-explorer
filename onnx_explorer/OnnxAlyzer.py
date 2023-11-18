import csv
import os
import json
import onnx
from collections import defaultdict
from onnx import numpy_helper

from onnx_explorer import logo_str


class ONNXModelAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def get_dtype_name(tensor_dtype):
        '''
        get data type name
        :param tensor_dtype:
        :return:
        '''
        return onnx.TensorProto.DataType.Name(tensor_dtype)

    @staticmethod
    def save_format_txt(output_info, output_file):
        with open(output_file + ".txt", "w") as f:
            f.write(f"{logo_str}\n")
            f.write("================summary================\n")
            for key, value in output_info["summary"].items():
                f.write(f"| {key}: {value}\n")

            f.write("=====parameter_data_types=====\n")
            for key, value in output_info["parameter_data_types"].items():
                f.write(f"| {key}: {value}\n")

            f.write("===========operators===========\n")
            for key, value in output_info["operators"].items():
                f.write(f"| {key}: count={value['count']}, percentage={value['percentage']}\n")

            f.write("===========inputs==============\n")
            for input_info in output_info["inputs"]:
                f.write(f"| name={input_info['name']}, dtype={input_info['dtype']}, shape={input_info['shape']}\n")

            f.write("===========outputs=============\n")
            for output_info in output_info["outputs"]:
                f.write(f"name={output_info['name']}, dtype={output_info['dtype']}, shape={output_info['shape']}\n")

            if "node_details" in output_info:
                f.write("=========node_details==========\n")
                for node_detail in output_info["node_details"]:
                    f.write(f"op_type={node_detail['op_type']}, name={node_detail['name']}\n")
                    f.write(f"inputs: {', '.join(node_detail['inputs'])}\n")
                    f.write(f"outputs: {', '.join(node_detail['outputs'])}\n")
                    f.write("attributes:\n")
                    for attr_name, attr_value in node_detail["attributes"].items():
                        f.write(f"  {attr_name}: {attr_value}\n")
                    f.write("\n")
        print(f"Model analysis saved to {output_file}.txt")

    @staticmethod
    def save_format_json(output_file, output_info):
        with open(output_file + ".json", "w") as f:
            json.dump(output_info, f, indent=2)
        print(f"\nModel analysis saved to {output_file}.json")

    @staticmethod
    def save_format_csv(output_info, output_file):
        with open(output_file + ".csv", "w", newline='') as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["section", "key", "value"])

            for key, value in output_info["summary"].items():
                csv_writer.writerow(["summary", key, value])

            for key, value in output_info["parameter_data_types"].items():
                csv_writer.writerow(["parameter_data_types", key, value])

            for key, value in output_info["operators"].items():
                csv_writer.writerow(["operators", key, f"count={value['count']}, percentage={value['percentage']}"])

            for input_info in output_info["inputs"]:
                csv_writer.writerow(
                    ["inputs", input_info['name'], f"dtype={input_info['dtype']}, shape={input_info['shape']}"])

            for output_info in output_info["outputs"]:
                csv_writer.writerow(
                    ["outputs", output_info['name'], f"dtype={output_info['dtype']}, shape={output_info['shape']}"])

            if "node_details" in output_info:
                for node_detail in output_info["node_details"]:
                    csv_writer.writerow(["node_details", node_detail['name'], f"op_type={node_detail['op_type']}"])
                    csv_writer.writerow(
                        ["node_details", node_detail['name'], f"inputs: {', '.join(node_detail['inputs'])}"])
                    csv_writer.writerow(
                        ["node_details", node_detail['name'], f"outputs: {', '.join(node_detail['outputs'])}"])
                    for attr_name, attr_value in node_detail["attributes"].items():
                        csv_writer.writerow(["node_details", node_detail['name'], f"{attr_name}: {attr_value}"])
        print(f"Model analysis saved to {output_file}.csv")

    @staticmethod
    def analyze_onnx_model(onnx_file_path, save_to_file=False, output_file=None, show_node_details=False):
        '''
        analyze onnx model
        :param onnx_file_path:
        :param save_to_file:
        :param output_file:
        :param show_node_details:
        :return:
        '''
        # Load ONNX model
        model = onnx.load(onnx_file_path)

        # Validate model
        onnx.checker.check_model(model)

        # Get graph information
        graph = model.graph

        # Get node information
        nodes = graph.node

        # Get input and output tensor information
        inputs = graph.input
        outputs = graph.output

        # Get model parameters
        initializer = graph.initializer

        # Count the number of nodes
        node_count = len(nodes)

        # Count the number of input and output tensors
        input_count = len(inputs)
        output_count = len(outputs)

        # Calculate the number of parameters
        num_params = sum(numpy_helper.to_array(tensor).size for tensor in initializer)

        # Count the number of operators
        op_count = defaultdict(int)
        for node in nodes:
            op_count[node.op_type] += 1

        # Calculate operator percentage
        op_percentage = {op_type: count / node_count * 100 for op_type, count in op_count.items()}

        # Calculate model size
        model_size = os.path.getsize(onnx_file_path)

        # Count the number of parameters for each data type
        dtype_count = defaultdict(int)
        for tensor in initializer:
            dtype_name = ONNXModelAnalyzer.get_dtype_name(tensor.data_type)
            dtype_count[dtype_name] += numpy_helper.to_array(tensor).size

        # Prepare output information
        output_info = {
            "summary": {
                "model": onnx_file_path,
                "node_count": node_count,
                "input_count": input_count,
                "output_count": output_count,
                "num_params": num_params,
                "model_size": model_size
            },
            "parameter_data_types": {dtype_name: count for dtype_name, count in dtype_count.items()},
            "operators": {op_type: {"count": count, "percentage": op_percentage[op_type]} for op_type, count in
                          op_count.items()},
            "inputs": [{"name": input_tensor.name,
                        "dtype": ONNXModelAnalyzer.get_dtype_name(input_tensor.type.tensor_type.elem_type),
                        "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]} for input_tensor in
                       inputs],
            "outputs": [{"name": output_tensor.name,
                         "dtype": ONNXModelAnalyzer.get_dtype_name(output_tensor.type.tensor_type.elem_type),
                         "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]} for output_tensor
                        in outputs], }

        if show_node_details:
            output_info["node_details"] = [
                {
                    "op_type": node.op_type,
                    "name": node.name,
                    "inputs": [input_name for input_name in node.input],
                    "outputs": [output_name for output_name in node.output],
                    "attributes": {attr.name: str(attr) for attr in node.attribute}
                } for node in nodes
            ]

            # Print output information
            import pprint
            pprint.pprint(output_info)

        # Save output information to file
        if save_to_file:
            if output_file is None:
                output_file = os.path.splitext(onnx_file_path)[0] + "_analysis"
            else:
                output_path = os.path.split(output_file)[0]
                print("output_path:", output_path)
                if os.path.exists(output_path) is False:
                    os.makedirs(output_path)

            # Save as JSON
            ONNXModelAnalyzer.save_format_json(output_file, output_info)

            # Save as TXT
            ONNXModelAnalyzer.save_format_txt(output_info, output_file)

            # Save as CSV
            ONNXModelAnalyzer.save_format_csv(output_info, output_file)




if __name__ == '__main__':
    # Usage example
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_file = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_analysis"
    ONNXModelAnalyzer.analyze_onnx_model(model_path, save_to_file=True, output_file=output_file, show_node_details=True)