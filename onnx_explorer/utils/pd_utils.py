import pandas as pd

# 读取CSV文件
data = pd.read_csv('/Users/gatilin/PycharmProjects/onnx-easy-tools/weights/yolov5/yolov5x6.csv')

# 将DataFrame转换为字符串
data_string = data.to_string()

# 完整地打印数据
print(data_string)