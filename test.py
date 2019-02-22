from conv import Conv
from layers import Layers
from utils import utils
import numpy as np

#path = 'lena512color.tiff'
#img = utils.read_image(path)

'''
img = np.random.random_sample((250, 250, 3))
cimg = Conv.conv(img, kernel_shape=(5, 5, 3), stride=5, padding=5,
        non_linear_func=lambda x: x)
print(cimg)
cimg = Conv.pool(cimg, filter_size=(5, 5), stride=5, pooling_function=np.average)
print(cimg)
'''

img_volume = np.random.random_sample((10, 250, 250, 3))
cimg = Layers.conv_layer(img_volume, kernel_shape=(5, 5, 3), stride=5, padding=5,
        non_linear_func=lambda x: x)
cimg = Layers.pool_layer(cimg, filter_size=(5, 5), stride=5, pooling_function=np.average)

print(cimg)
#utils.save_image('test_conv_lena512color.tiff', cimg)
