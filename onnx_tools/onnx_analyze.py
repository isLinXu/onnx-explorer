import os
import json
import onnx
from collections import defaultdict
from onnx import numpy_helper

def get_dtype_name(tensor_dtype):
    '''
    get data type name
    :param tensor_dtype:
    :return:
    '''
    return onnx.TensorProto.DataType.Name(tensor_dtype)

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
        dtype_name = get_dtype_name(tensor.data_type)
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
        "operators": {op_type: {"count": count, "percentage": op_percentage[op_type]} for op_type, count in op_count.items()},
        "inputs": [{"name": input_tensor.name, "dtype": get_dtype_name(input_tensor.type.tensor_type.elem_type), "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]} for input_tensor in inputs],
        "outputs": [{"name": output_tensor.name, "dtype": get_dtype_name(output_tensor.type.tensor_type.elem_type), "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]} for output_tensor in outputs],
    }

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
        with open(output_file + ".json", "w") as f:
            json.dump(output_info, f, indent=2)
        print(f"\nModel analysis saved to {output_file}.json")

        # Save as TXT
        with open(output_file + ".txt", "w") as f:
            pprint.pprint(output_info, stream=f)
        print(f"Model analysis saved to {output_file}.txt")

if __name__ == '__main__':
    # Usage example
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    output_file = "/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov3/yolov3_analysis"
    analyze_onnx_model(model_path, save_to_file=True, output_file=output_file, show_node_details=True)