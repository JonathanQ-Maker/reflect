import urllib.request
import os.path

def to_tuple(x):
    """
    returns x if x is a tuple, else return (x, )
    """
    return x if isinstance(x, tuple) else (x, )

def conv_size(size: int, kernel_size: int, stride: int=1):
    """
    returns size of matrix after convolution over a single axis

    size:           size of input over axis 
    kernel_size:    size of kernel over axis
    stride:         convolution stride over axis
    """
    return int((size - kernel_size) / stride + 1)

def in_conv_size(out: int, kernel_size: int, stride: int=1):
    """
    returns size of matrix nessesary to get 
        out size after convolution over axis

    out:            out size over axis 
    kernel_size:    size of kernel over axis
    stride:         convolution stride over axis
    """
    return (out - 1) * stride + kernel_size

def print_progress(prefix, current, total):
    """
    prints progress bar of the format
        
        {prefix}: [==        ] 20.0%

    Args:
        prefix:     progress bar prefix
        current:    current progress (int)
        total:      total expected progress (int)

    NOTE: no other print is expected inbetween progress update
    """

    end = "\r"
    if (current == total):
        end = "\n"
    percent = round(current * 100.0 / total, 2)
    print((f"{prefix}: [{'='*current}{' '*(total - current)}] {percent}%"), end=end)

def download_file(url, location, filename, show=False):
    """
    downloads file from url and place into location with filename 
    
    Args:
        url:        file url
        location:   location to place file
        filename:   name given to downloaded file
        show:       show progress bar
    """

    def show_progress(block_num, block_size, total_size):
        print_progress(filename, block_num * block_size, total_size)

    print(f"Downloading {filename} from {url}")
    urllib.request.urlretrieve(url, os.path.join(location, filename), show_progress)