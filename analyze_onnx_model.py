import os
import onnx
from collections import defaultdict
from onnx import numpy_helper

def get_dtype_name(tensor_dtype):
    return onnx.TensorProto.DataType.Name(tensor_dtype)

def analyze_onnx_model(onnx_file_path, save_to_file=False, output_file=None):
    # 加载ONNX模型
    model = onnx.load(onnx_file_path)

    # 获取图信息
    graph = model.graph

    # 获取节点信息
    nodes = graph.node

    # 获取输入和输出张量信息
    inputs = graph.input
    outputs = graph.output

    # 获取模型参数
    initializer = graph.initializer

    # 统计节点数量
    node_count = len(nodes)

    # 统计输入和输出张量数量
    input_count = len(inputs)
    output_count = len(outputs)

    # 计算参数量
    num_params = sum(numpy_helper.to_array(tensor).size for tensor in initializer)

    # 统计算子数量
    op_count = defaultdict(int)
    for node in nodes:
        op_count[node.op_type] += 1

    # 计算算子百分比
    op_percentage = {op_type: count / node_count * 100 for op_type, count in op_count.items()}

    # 计算模型大小
    model_size = os.path.getsize(onnx_file_path)

    # 准备输出信息
    output_info = []

    # 添加摘要信息
    output_info.append(f"Model: {onnx_file_path}")
    output_info.append(f"Node count: {node_count}")
    output_info.append(f"Input count: {input_count}")
    output_info.append(f"Output count: {output_count}")
    output_info.append(f"Number of parameters: {num_params}")
    output_info.append(f"Model size: {model_size} bytes")

    # 添加算子信息
    output_info.append("\nOperators:")
    for i, (op_type, count) in enumerate(op_count.items()):
        output_info.append(f"{i + 1}. {op_type} (count: {count}, percentage: {op_percentage[op_type]:.2f}%)")

    # 添加输入信息
    output_info.append("\nInputs:")
    for i, input_tensor in enumerate(inputs):
        dtype_name = get_dtype_name(input_tensor.type.tensor_type.elem_type)
        output_info.append(f"{i + 1}. {input_tensor.name} (dtype: {dtype_name}, shape: {list(input_tensor.type.tensor_type.shape.dim)})")

    # 添加输出信息
    output_info.append("\nOutputs:")
    for i, output_tensor in enumerate(outputs):
        dtype_name = get_dtype_name(output_tensor.type.tensor_type.elem_type)
        output_info.append(f"{i + 1}. {output_tensor.name} (dtype: {dtype_name}, shape: {list(output_tensor.type.tensor_type.shape.dim)})")

    # 添加节点详细信息
    output_info.append("\nNode Details:")
    for i, node in enumerate(nodes):
        output_info.append(f"{i + 1}. {node.op_type} ({node.name})")
        output_info.append(f"  Inputs: {len(node.input)}")
        for input_name in node.input:
            output_info.append(f"    {input_name}")
        output_info.append(f"  Outputs: {len(node.output)}")
        for output_name in node.output:
            output_info.append(f"    {output_name}")
        # 添加属性信息
        output_info.append("  Attributes:")
        for attr in node.attribute:
            output_info.append(f"    {attr.name}: {attr}")

    # 打印输出信息
    for line in output_info:
        print(line)

    # 保存输出信息到文件
    if save_to_file:
        if output_file is None:
            output_file = os.path.splitext(onnx_file_path)[0] + "_analysis.txt"
        with open(output_file, "w") as f:
            for line in output_info:
                f.write(line + "\n")
        print(f"\nModel analysis saved to {output_file}")

if __name__ == '__main__':
    # 使用示例
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    analyze_onnx_model(model_path, save_to_file=True)