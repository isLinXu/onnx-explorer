import os
import re
import csv
from itertools import groupby

class SummaryExtractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.summaries = []

    def extract_summaries(self):
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()

                        model = re.search(r'model: (.*)', content).group(1)
                        node_count = int(re.search(r'node_count: (\d+)', content).group(1))
                        input_count = int(re.search(r'input_count: (\d+)', content).group(1))
                        output_count = int(re.search(r'output_count: (\d+)', content).group(1))
                        num_params = int(re.search(r'num_params: (\d+)', content).group(1))
                        model_size = float(re.search(r'model_size: ([\d.]+)', content).group(1))

                        self.summaries.append([file, model, node_count, input_count, output_count, num_params, model_size])

    def sort_summaries(self):
        sorted_summaries_by_model = sorted(self.summaries, key=lambda x: x[1])
        sorted_summaries = []

        for _, group in groupby(sorted_summaries_by_model, key=lambda x: x[1]):
            sorted_group = sorted(group, key=lambda x: x[6])
            sorted_summaries.extend(sorted_group)

        self.summaries = sorted_summaries

    def save_to_csv(self, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File Name', 'Model', 'Node Count', 'Input Count', 'Output Count', 'Num Params', 'Model Size'])
            csv_writer.writerows(self.summaries)

    def save_to_md(self, file_name):
        with open(file_name, 'w') as mdfile:
            mdfile.write('| File Name | Model | Node Count | Input Count | Output Count | Num Params | Model Size |\n')
            mdfile.write('|-----------|-------|------------|-------------|--------------|------------|------------|\n')

            for summary in self.summaries:
                mdfile.write('| {} | {} | {} | {} | {} | {} | {} |\n'.format(*summary))

if __name__ == '__main__':
    folder_path = 'path/to/txt/files'
    extractor = SummaryExtractor(folder_path)
    extractor.extract_summaries()
    extractor.sort_summaries()
    extractor.save_to_csv('sorted_summary.csv')
    extractor.save_to_md('sorted_summary.md')