import os

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