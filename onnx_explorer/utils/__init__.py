import os

import onnx
from onnx import numpy_helper

# def get_file_size(file_path):
#     try:
#         file_info = os.stat(file_path)
#         return file_info.st_blocks * file_info.st_blksize
#     except OSError:
#         return 0

def get_file_size_mb(file_path):
    try:
        file_info = os.stat(file_path)
        size_in_bytes = file_info.st_size
        size_in_mb = size_in_bytes / (1024 * 1024)
        return size_in_mb
    except OSError:
        return 0

import numpy as np
from onnx import numpy_helper

def get_model_size_mb(model_path):
    # Load ONNX model
    model = onnx.load(model_path)

    # Get model parameters
    initializer = model.graph.initializer

    # Calculate the total number of parameters
    total_params = sum(np.prod(numpy_helper.to_array(tensor).shape) for tensor in initializer)

    # Calculate the total size in bytes (each parameter is a float32, i.e., 4 bytes)
    total_size_bytes = total_params * 4

    # Convert to MB
    total_size_mb = total_size_bytes / (1024 * 1024)

    return total_size_mb

def byte_to_mb(byte):
    return byte / ( 1024 * 1024)

def bytes_to_readable_size(size_in_bytes, unit='MB'):
    if unit == 'MB':
        return size_in_bytes / (1024 * 1024)
    elif unit == 'KB':
        return size_in_bytes / 1024
    elif unit == 'GB':
        return size_in_bytes / (1024 * 1024 * 1024)
    else:
        raise ValueError("Invalid unit. Supported units are 'MB' and 'KB'.")

def get_file_size(file_path,unit='MB'):
    file_path_size = os.path.getsize(file_path)
    file_size_mb = bytes_to_readable_size(file_path_size, unit=unit)
    return file_size_mb