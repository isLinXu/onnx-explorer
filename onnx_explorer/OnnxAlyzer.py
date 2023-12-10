import csv
import os
import json

import numpy
import numpy as np
import onnx
from collections import defaultdict
from onnx import numpy_helper
from onnx import shape_inference
from onnx_explorer import logo_str
from onnx_explorer.utils import get_file_size, byte_to_mb, get_file_size_mb, get_model_size_mb


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
    def save_format_txt(output_info, output_file, show_node_details=False):
        '''
        :param output_info:
        :param output_file:
        :return:
        1.summary
        2.parameter_data_types
        3.operators-lists
        4.operators
        5.inputs
        6.outputs
        7.node_details
        '''
        with open(output_file + ".txt", "w") as f:
            f.write(f"{logo_str}\n")
            # Write model information in the given format
            if show_node_details:
                # Calculate input size (MB)
                input_size = sum(numpy.prod(input_info['shape']) for input_info in output_info['inputs']) * 4 / (1024 * 1024)

                forward_backward_pass_size = sum(numpy.prod(node_detail['output_shape']) for node_detail in output_info['node_details']) * 4 / (1024 * 1024)

                # Calculate estimated total size (MB)
                estimated_total_size = input_size + forward_backward_pass_size + output_info['summary']['model_size']

                # Write model information in the given format
                f.write("=========================================================================================================\n")
                f.write("Layer (type:depth-idx)                                  Output Shape              Param #\n")
                f.write("=========================================================================================================\n")

                if "node_details" in output_info:
                    sorted_node_details = sorted(output_info.get("node_details", []), key=lambda x: x['depth'])
                    last_depth = -1
                    for node_detail in sorted_node_details:
                        output_shape = str(node_detail['output_shape'])
                        param_num = node_detail['param_count']
                        depth = node_detail['depth']
                        layer_name = node_detail['name']
                        if depth > last_depth:
                            indent = '│  ' * depth
                            branch = '├─'
                        elif depth == last_depth:
                            indent = '│  ' * (depth - 1)
                            branch = '├─'
                        else:  # depth < last_depth
                            indent = '│  ' * depth
                            branch = '└─'
                        last_depth = depth
                        op_type_str = f"{node_detail['op_type']} (d={depth}):"
                        param_num_str = f"{param_num: <6}"
                        f.write(f"{indent}{branch}{op_type_str: <18} {layer_name: <20} {output_shape: <25} {param_num_str}\n")
                f.write("=========================================================================================================\n")
                f.write(f"Total params: {output_info['summary']['num_params']}\n")
                f.write(f"Trainable params: {output_info['summary']['num_params']}\n")
                f.write("Non-trainable params: 0\n")
                f.write(
                    "=========================================================================================================\n")
                f.write(f"Input size (MB): {input_size:.2f}\n")
                f.write(f"Forward/backward pass size (MB): {forward_backward_pass_size:.2f}\n")
                f.write(f"Params size (MB): {output_info['summary']['model_size']}\n")
                f.write(f"Estimated Total Size (MB): {estimated_total_size:.2f}\n")
                f.write(
                    "=========================================================================================================\n\n")

            ############################################################################################################################
            # Write model information in the given format
            f.write("================================【summary】================================\n")
            for key, value in output_info["summary"].items():
                f.write(f"| {key}: {value}\n")

            f.write("=====================【parameter_data_types】=====================\n")
            for key, value in output_info["parameter_data_types"].items():
                f.write(f"| {key}: {value}\n")

            f.write("===========================【operators-lists】===========================\n")
            operators_list = output_info["operators_list"]
            f.write(f"| {operators_list}\n")

            f.write("===========================【operators】===========================\n")
            for key, value in output_info["operators"].items():
                f.write(f"| {key}: count={value['count']}, percentage={value['percentage']}\n")

            f.write("===========================【inputs】==============================\n")
            for input_info in output_info["inputs"]:
                f.write(f"| name={input_info['name']}, dtype={input_info['dtype']}, shape={input_info['shape']}\n")

            f.write("===========================【outputs】=============================\n")
            for output_info in output_info["outputs"]:
                f.write(f"name={output_info['name']}, dtype={output_info['dtype']}, shape={output_info['shape']}\n")

            if "node_details" in output_info:
                f.write(
                    "=========================================【node_details】==========================================\n")
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
    def get_file_size(file_path):
        return os.path.getsize(file_path)
    @staticmethod
    def get_node_depth(node_name, node_parents, nodes_by_output, visited=None):
        '''
        get node depth
        :param node_name:
        :param node_parents:
        :param nodes_by_output:
        :param visited:
        :return:
        '''
        if visited is None:
            visited = set()
        if node_name in visited:
            return 0
        visited.add(node_name)
        depth = 0
        if node_name in node_parents:
            parent_depths = [ONNXModelAnalyzer.get_node_depth(parent_name, node_parents, nodes_by_output, visited) for
                             parent_name in node_parents[node_name]]
            depth = max(parent_depths) + 1
        return depth

    @staticmethod
    def print_node_structure(f, node_name, node_details, node_parent, depth=0):
        '''
        print node structure
        :param f:
        :param node_name:
        :param node_details:
        :param node_parent:
        :param depth:
        :return:
        '''
        node_detail = node_details[node_name]
        f.write("─" * depth + f"{node_detail['op_type']} ({node_detail['name']})\n")
        for child_node_name in [node['name'] for node in node_details if node_parent[node['name']] == node_name]:
            f.print_node_structure(f, child_node_name, node_details, node_parent, depth + 1)

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
        if onnx_file_path is not None:
            if os.path.exists(onnx_file_path):
                # Load ONNX model
                model = onnx.load(onnx_file_path)

                # Validate model
                onnx.checker.check_model(model)

                # Infer shapes
                inferred_model = shape_inference.infer_shapes(model)
                value_info = {vi.name: vi for vi in inferred_model.graph.value_info}

                # Get graph information
                graph = model.graph

                # Get node information
                nodes = graph.node

                # Get input and output tensor information
                inputs = graph.input
                outputs = graph.output

                # Get model parameters
                initializer = graph.initializer

                # Calculate parameters for each node
                initializer_dict = {tensor.name: numpy_helper.to_array(tensor) for tensor in initializer}
                node_params = {}
                for node in nodes:
                    node_param_count = 0
                    for input_name in node.input:
                        if input_name in initializer_dict:
                            node_param_count += initializer_dict[input_name].size
                    node_params[node.name] = node_param_count

                # Create a dictionary to find nodes by their output tensor names
                nodes_by_output = {output_name: node for node in nodes for output_name in node.output}

                # Calculate parents for each node
                node_parents = defaultdict(list)
                for node in nodes:
                    for input_name in node.input:
                        if input_name in nodes_by_output:
                            node_parents[node.name].append(nodes_by_output[input_name].name)
                print("Node parents:")
                for node_name, parent_list in node_parents.items():
                    print(f"{node_name}: {parent_list}")
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

                # Count the number of parameters for each data type
                dtype_count = defaultdict(int)
                for tensor in initializer:
                    dtype_name = ONNXModelAnalyzer.get_dtype_name(tensor.data_type)
                    dtype_count[dtype_name] += numpy_helper.to_array(tensor).size

                nodes_by_name = {node.name: node for node in nodes}

                # Calculate model size
                # model_size = os.path.getsize(onnx_file_path)
                # model_size = get_file_size_mb(onnx_file_path)
                model_size = get_model_size_mb(onnx_file_path)
                # Prepare output information
                output_info = {
                    "summary": {
                        "model": onnx_file_path,
                        "node_count": node_count,
                        "input_count": input_count,
                        "output_count": output_count,
                        "num_params": num_params,
                        "model_size": model_size,
                        # "model_size": byte_to_mb(model_size)
                    },
                    "parameter_data_types": {dtype_name: count for dtype_name, count in dtype_count.items()},
                    "operators": {op_type: {"count": count, "percentage": op_percentage[op_type]} for op_type, count in
                                  op_count.items()}, "operators_list": list(op_count.keys()),
                    "inputs": [{"name": input_tensor.name,
                                "dtype": ONNXModelAnalyzer.get_dtype_name(input_tensor.type.tensor_type.elem_type),
                                "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]} for input_tensor in inputs],
                                "outputs": [{"name": output_tensor.name,
                                             "dtype": ONNXModelAnalyzer.get_dtype_name(output_tensor.type.tensor_type.elem_type),
                                             "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]} for output_tensor in outputs],
                                "node_details": [
                                {
                                    "op_type": node.op_type,
                                    "name": node.name,
                                    "inputs": [input_name for input_name in node.input],
                                    "outputs": [output_name for output_name in node.output],
                                    "attributes": {attr.name: str(attr) for attr in node.attribute},
                                    "output_shape": [dim.dim_value for dim in value_info[node.output[0]].type.tensor_type.shape.dim] if node.output[0] in value_info else [],
                                    "param_count": node_params[node.name],
                                    "depth": ONNXModelAnalyzer.get_node_depth(node.name, node_parents, nodes_by_output)
                                } for node in nodes
                            ]}

                if show_node_details:
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
                    ONNXModelAnalyzer.save_format_txt(output_info, output_file, show_node_details)

                    # Save as CSV
                    ONNXModelAnalyzer.save_format_csv(output_info, output_file)
                else:
                    print("onnx_file_path not found")
            else:
                print("onnx_file_path is None")


def main():
    print(f"{logo_str}\n")
    model_path = "../ckpts/yolov5/yolov5x6.onnx"
    output_file = "../weights/yolov5/yolov5x6"
    ONNXModelAnalyzer.analyze_onnx_model(model_path, save_to_file=True, output_file=output_file,show_node_details=False)

if __name__ == '__main__':
    main()
