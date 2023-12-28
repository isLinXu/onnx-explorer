import os
import re
import csv

# 修改为你的txt文件所在的文件夹路径
folder_path = '/Users/gatilin/PycharmProjects/onnx-easy-tools/infos/detectron2/'

# 创建一个列表，用于存储summary信息
summaries = []

# 遍历文件夹及其子文件夹中的所有txt文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.txt'):
            print("file: ", file)
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()

                # 使用正则表达式提取summary信息
                file = file.split(".")[0]
                model = re.search(r'model: (.*)', content).group(1)
                model = model.split("/")[-1]
                node_count = int(re.search(r'node_count: (\d+)', content).group(1))
                input_count = int(re.search(r'input_count: (\d+)', content).group(1))
                output_count = int(re.search(r'output_count: (\d+)', content).group(1))
                num_params = int(re.search(r'num_params: (\d+)', content).group(1))
                model_size = float(re.search(r'model_size: ([\d.]+)', content).group(1))


                # 将提取到的信息添加到列表中
                summaries.append([file, model, node_count, input_count, output_count, num_params, model_size])

# 根据模型名称和Model Size对summary信息进行排序
sorted_summaries = sorted(summaries, key=lambda x: (x[1], x[6]))

# 将summary信息保存到CSV文件中
with open('summary.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Name', 'Model', 'Node Count', 'Input Count', 'Output Count', 'Num Params', 'Model Size'])
    csv_writer.writerows(summaries)

# 将summary信息插入到Markdown文件中
with open('summary.md', 'w') as mdfile:
    mdfile.write('| File Name | Model | Node Count | Input Count | Output Count | Num Params | Model Size |\n')
    mdfile.write('|-----------|-------|------------|-------------|--------------|------------|------------|\n')

    for summary in summaries:
        mdfile.write('| {} | {} | {} | {} | {} | {} | {} |\n'.format(*summary))