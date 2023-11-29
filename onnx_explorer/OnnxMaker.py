

import yaml
import torch
import torch.nn as nn
import onnx

class CustomModel(nn.Module):
    def __init__(self, layers):
        super(CustomModel, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, (Concat, Add)):
                inputs = [outputs[input_name] for input_name in layer.inputs]
                x = layer(*inputs)
            else:
                x = layer(x)

            outputs[f"layer{idx}"] = x

        return x


class Concat(nn.Module):
    def __init__(self, inputs, dim):
        super(Concat, self).__init__()
        self.inputs = inputs
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class Add(nn.Module):
    def __init__(self, inputs):
        super(Add, self).__init__()
        self.inputs = inputs

    def forward(self, *inputs):
        return sum(inputs)


def create_pytorch_layer(layer_type, layer_params):
    # ... 添加其他层类型 ...
    if layer_type == "Linear":
        return nn.Linear(layer_params["in_features"], layer_params["out_features"])
    elif layer_type == "ReLU":
        return nn.ReLU()
    elif layer_type == "Conv2d":
        return nn.Conv2d(layer_params["in_channels"], layer_params["out_channels"], layer_params["kernel_size"],
                         stride=layer_params.get("stride", 1), padding=layer_params.get("padding", 0))
    elif layer_type == "MaxPool2d":
        return nn.MaxPool2d(layer_params["kernel_size"], stride=layer_params.get("stride", None),
                            padding=layer_params.get("padding", 0))
    elif layer_type == "BatchNorm2d":
        return nn.BatchNorm2d(layer_params["num_features"])
    elif layer_type == "Dropout":
        return nn.Dropout(layer_params.get("p", 0.5))
    elif layer_type == "Dropout2d":
        return nn.Dropout2d(layer_params.get("p", 0.5))
    elif layer_type == "AvgPool2d":
        return nn.AvgPool2d(layer_params["kernel_size"], stride=layer_params.get("stride", None),
                            padding=layer_params.get("padding", 0))
    elif layer_type == "AdaptiveAvgPool2d":
        return nn.AdaptiveAvgPool2d(layer_params["output_size"])
    elif layer_type == "AdaptiveMaxPool2d":
        return nn.AdaptiveMaxPool2d(layer_params["output_size"])
    elif layer_type == "Softmax":
        return nn.Softmax(dim=layer_params.get("dim", None))
    elif layer_type == "LogSoftmax":
        return nn.LogSoftmax(dim=layer_params.get("dim", None))
    elif layer_type == "Tanh":
        return nn.Tanh()
    elif layer_type == "Sigmoid":
        return nn.Sigmoid()
    elif layer_type == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=layer_params.get("negative_slope", 0.01))
    elif layer_type == "ELU":
        return nn.ELU(alpha=layer_params.get("alpha", 1.0))
    elif layer_type == "PReLU":
        return nn.PReLU(num_parameters=layer_params.get("num_parameters", 1), init=layer_params.get("init", 0.25))
    elif layer_type == "Softplus":
        return nn.Softplus(beta=layer_params.get("beta", 1), threshold=layer_params.get("threshold", 20))
    elif layer_type == "Softshrink":
        return nn.Softshrink(lambd=layer_params.get("lambd", 0.5))
    elif layer_type == "Softmin":
        return nn.Softmin(dim=layer_params.get("dim", None))
    elif layer_type == "Hardtanh":
        return nn.Hardtanh(min_val=layer_params.get("min_val", -1.0), max_val=layer_params.get("max_val", 1.0))
    # 添加更多的层类型
    else:
        if layer_type == "Concat":
            return Concat(layer_params["inputs"], layer_params["dim"])
        elif layer_type == "Add":
            return Add(layer_params["inputs"])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


def create_pytorch_model_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        model_structure = yaml.safe_load(f)

    layers = []
    for layer in model_structure["layers"]:
        layer_type = layer["type"]
        layer_params = {k: v for k, v in layer.items() if k != "type"}
        layers.append(create_pytorch_layer(layer_type, layer_params))

    return CustomModel(layers)


def export_pytorch_model_to_onnx(pytorch_model, onnx_path, input_shape):
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(pytorch_model, dummy_input, onnx_path)


yaml_path = "/Users/gatilin/PycharmProjects/onnx-easy-tools/config/model_structure.yaml"
pytorch_model = create_pytorch_model_from_yaml(yaml_path)
onnx_path = "model_structure.onnx"
input_shape = (1, 3)
export_pytorch_model_to_onnx(pytorch_model, onnx_path, input_shape)
