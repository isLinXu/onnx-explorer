<img width="880" alt="onnx-explorer-logo" src="https://github.com/isLinXu/issues/assets/59380685/79a069e3-6a22-4a33-9d87-b343b7eadcdf">



# Design


![onnx-explorer-design](https://github.com/isLinXu/issues/assets/59380685/ad11988b-3ca3-4014-83a9-66cf952beb07)

## Features
- [x]  support export onnx model infos and summary[txt,json,csv]
- [x] support estimate model memory
- [x] support onnx model infer

# Example

## onnx model infos

| <img width="1005" alt="onnx_explorer_3" src="https://github.com/isLinXu/issues/assets/59380685/18c3ea72-2c81-4ff0-b8f1-c993c9020967"> | <img width="760" alt="onnx_explorer_4" src="https://github.com/isLinXu/issues/assets/59380685/e2cb1947-aff6-4552-8150-b3a7a8a2e59b"> | <img width="1034" alt="image" src="https://github.com/isLinXu/issues/assets/59380685/e2613ae9-6890-4b58-ac05-64222bbfefc2"> |
|:-------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|
|  [Yolov5x6.txt](example/data/yolov5/yolov5x6.txt)                                                            | [Yolov5x6.json](example/data/yolov5/yolov5x6.json)                                                           | [Yolov5x6.csv](example/data/yolov5/yolov5x6.csv)                                                       |

##  estimate model memory

| <img width="929" alt="onnx-explorer_2" src="https://github.com/isLinXu/issues/assets/59380685/490fef9b-2268-4583-8a91-53005643f447"> |
| ------------------------------------------------------------ |
|                                                              |


## onnx model infer

| <img width="802" alt="onnx-infer" src="https://github.com/isLinXu/issues/assets/59380685/4c1b6e9a-8b7b-4ccb-bb15-4ee7ca19645a"> |
| ------------------------------------------------------------ |
|                                                              |


# Usage

## Install

```shell
conda create -n onnx-explorer python=3.8
conda activate onnx-explorer
python3 -m pip install -r requirements.txt
```

onnx-explorer building...

## Easy start run

```shell
sh run.sh
```

```shell
sh run_estimate.sh
```

```shell
sh infer.sh
```
