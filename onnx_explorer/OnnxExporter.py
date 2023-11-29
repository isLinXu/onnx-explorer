import torch
import torch.onnx
import argparse
import os
import logging
from torchvision import models

logging.basicConfig(level=logging.INFO)

def convert_pt_to_onnx(pt_path, onnx_path, input_size, model_name=None):
    # 加载PyTorch模型
    if model_name is not None:
        model = getattr(models, model_name)(pretrained=True)
    else:
        if not os.path.exists(pt_path):
            logging.error(f"{pt_path} does not exist.")
            return
        try:
            model = torch.load(pt_path)
        except Exception as e:
            logging.error(f"Error loading {pt_path}: {e}")
            return
    model.eval()

    # 创建一个虚拟输入，以便ONNX导出器可以确定输入形状
    dummy_input = torch.randn(1, *input_size)

    # 导出模型
    try:
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
        logging.info(f"Model converted from {pt_path} to {onnx_path}")
    except Exception as e:
        logging.error(f"Error converting {pt_path} to {onnx_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt to .onnx")
    parser.add_argument("--pt_path", type=str, help="Path to .pt file", default=None)
    parser.add_argument("--onnx_path", type=str, help="Path to output .onnx file")
    parser.add_argument("--input_size", type=int, nargs="+", help="Input size of the model (C, H, W)")
    parser.add_argument("--model_name", type=str, help="Name of the torchvision model (if using a pretrained model)", default=None)

    args = parser.parse_args()
    convert_pt_to_onnx(args.pt_path, args.onnx_path, tuple(args.input_size), args.model_name)