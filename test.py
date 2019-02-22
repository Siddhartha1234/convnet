from conv import Conv
from utils import utils
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

path = 'lena512color.tiff'
img = utils.read_image(path)
cimg = Conv.conv(img, kernel_shape=(5, 5, 3), stride=5, padding=5,
        non_linear_func=sigmoid)
cimg = Conv.pool(cimg, filter_size=(5,5), stride=5, pooling_function=np.max)
utils.save_image('test_conv_lena512color.tiff', cimg)
