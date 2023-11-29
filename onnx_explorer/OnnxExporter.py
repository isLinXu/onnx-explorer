import torch
import torch.onnx
import argparse

def convert_pt_to_onnx(pt_path, onnx_path, input_size):
    # 加载PyTorch模型
    model = torch.load(pt_path)
    model.eval()

    # 创建一个虚拟输入，以便ONNX导出器可以确定输入形状
    dummy_input = torch.randn(1, *input_size)

    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model converted from {pt_path} to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt to .onnx")
    parser.add_argument("--pt_path", type=str, help="Path to .pt file")
    parser.add_argument("--onnx_path", type=str, help="Path to output .onnx file")
    parser.add_argument("--input_size", type=int, nargs="+", help="Input size of the model (C, H, W)")

    args = parser.parse_args()
    convert_pt_to_onnx(args.pt_path, args.onnx_path, tuple(args.input_size))