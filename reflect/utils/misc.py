
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