from tests import DATA_FOLDER_PATH
from reflect.utils.misc import download_file
import os.path
import gzip, pickle
import numpy as np

"""
MNIST dataset utilities

URL: http://yann.lecun.com/exdb/mnist/
"""

# MNIST urls
TRAIN_IMAGES_URL    = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS_URL    = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGES_URL     = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_URL     = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

# MNIST filenames
TRAIN_IMAGES    = "train-images-idx3-ubyte.gz"
TRAIN_LABELS    = "train-labels-idx1-ubyte.gz"
TEST_IMAGES     = "t10k-images-idx3-ubyte.gz"
TEST_LABELS     = "t10k-labels-idx1-ubyte.gz"

IMG_DIM = 28
IMG_LENGTH = IMG_DIM * IMG_DIM

# decode methods adapted from https://stackoverflow.com/a/53448171
def decode_image_file(path, flat=True):
    """
    decodes MNIST image file

    Args:
        path: path to file

    Returns:
        numpy array of shape (# examples, 784) or (# examples, 28, 28, 1)
    """
    n_bytes_per_img = IMG_DIM*IMG_DIM

    with gzip.open(path, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[16:]

        if len(data) % n_bytes_per_img != 0:
            raise Exception('Something wrong with the file')
        result = np.frombuffer(data, dtype=np.uint8)
        if (flat):
            result.shape = (len(bytes_)//n_bytes_per_img, n_bytes_per_img)
        else:
            result.shape = (len(bytes_)//n_bytes_per_img, IMG_DIM, IMG_DIM, 1)
        return result

def decode_label_file(path):
    """
    decodes MNIST label file

    Args:
        path: path to file

    Returns:
        numpy array of shape (# examples, ) 
    """
    with gzip.open(path, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[8:]

        return np.frombuffer(data, dtype=np.uint8)


def load_MNIST_image(url, filename, flat=True):
    """
    load MNIST data and encode into images. 
    If MNIST file does not exist file will be downloaded from url

    Args:
        url:        url to download file from
        filename:   file to load
        flat:       should image be flattened

    Returns:
        numpy array of shape (# examples, 784) or (# examples, 28, 28, 1)
    """

    file_path = os.path.join(DATA_FOLDER_PATH, filename)
    if not os.path.isfile(file_path):
        # file does not exist
        download_file(url, DATA_FOLDER_PATH, filename, True)

    return decode_image_file(file_path, flat)

def load_MNIST_label(url, filename):
    """
    load MNIST data and encode into labels. 
    If MNIST file does not exist file will be downloaded from url

    Args:
        url:        url to download file from
        filename:   file to load

    Returns:
        numpy array of shape (# examples, ) 
    """
    file_path = os.path.join(DATA_FOLDER_PATH, filename)
    if not os.path.isfile(file_path):
        # file does not exist
        download_file(url, DATA_FOLDER_PATH, filename, True)

    return decode_label_file(file_path)