import onnx

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

    # 统计节点数量
    node_count = len(nodes)

    # 统计输入和输出张量数量
    input_count = len(inputs)
    output_count = len(outputs)

    # 打印摘要信息
    print(f"Model: {onnx_file_path}")
    print(f"Node count: {node_count}")
    print(f"Input count: {input_count}")
    print(f"Output count: {output_count}")
    print("\nNodes:")

    # 打印节点详细信息
    for i, node in enumerate(nodes):
        print(f"{i + 1}. {node.op_type} ({node.name})")

    print("\nInputs:")
    for i, input_tensor in enumerate(inputs):
        print(f"{i + 1}. {input_tensor.name} (shape: {list(input_tensor.type.tensor_type.shape.dim)})")

    print("\nOutputs:")
    for i, output_tensor in enumerate(outputs):
        print(f"{i + 1}. {output_tensor.name} (shape: {list(output_tensor.type.tensor_type.shape.dim)})")


if __name__ == '__main__':
    # 使用示例
    model_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov3/yolov3.onnx"
    analyze_onnx_model(model_path)