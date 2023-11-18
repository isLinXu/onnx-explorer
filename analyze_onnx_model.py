import onnx
from collections import defaultdict
from onnx import numpy_helper


def get_dtype_name(tensor_dtype):
    return onnx.TensorProto.DataType.Name(tensor_dtype)


def analyze_onnx_model(onnx_file_path):
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

    # 打印摘要信息
    print(f"Model: {onnx_file_path}")
    print(f"Node count: {node_count}")
    print(f"Input count: {input_count}")
    print(f"Output count: {output_count}")
    print(f"Number of parameters: {num_params}")
    print("\nOperators:")

    # 打印算子信息
    for i, (op_type, count) in enumerate(op_count.items()):
        print(f"{i + 1}. {op_type} (count: {count})")

    print("\nInputs:")
    for i, input_tensor in enumerate(inputs):
        dtype_name = get_dtype_name(input_tensor.type.tensor_type.elem_type)
        print(
            f"{i + 1}. {input_tensor.name} (dtype: {dtype_name}, shape: {list(input_tensor.type.tensor_type.shape.dim)})")

    print("\nOutputs:")
    for i, output_tensor in enumerate(outputs):
        dtype_name = get_dtype_name(output_tensor.type.tensor_type.elem_type)
        print(
            f"{i + 1}. {output_tensor.name} (dtype: {dtype_name}, shape: {list(output_tensor.type.tensor_type.shape.dim)})")

    # 打印节点详细信息
    print("\nNode Details:")
    for i, node in enumerate(nodes):
        print(f"{i + 1}. {node.op_type} ({node.name})")
        print(f"  Inputs: {len(node.input)}")
        for input_name in node.input:
            print(f"    {input_name}")
        print(f"  Outputs: {len(node.output)}")
        for output_name in node.output:
            print(f"    {output_name}")
        # 打印属性信息
        print("  Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name}: {attr}")

if __name__ == '__main__':
    # 使用示例
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    analyze_onnx_model(model_path)
