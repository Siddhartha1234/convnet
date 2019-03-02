from conv import Conv
from layers import Layers
from utils import utils
import numpy as np
from convnet import CNN

#path = 'lena512color.tiff'
#img = utils.read_image(path)

'''
c = Conv()
img = np.random.random_sample((250, 250, 3))
cimg = c.conv(img, kernel_shape=(5, 5, 3), stride=5, padding=5,
        non_linear_func=lambda x: x)
print(cimg)
cimg = c.pool(cimg, filter_size=(5, 5), stride=5, pooling_function=np.average)
print(cimg)
'''

'''
img_volume = np.random.random_sample((10, 400, 400))
lay = Layers()
cimg = lay.conv_layer(img_volume, kernel_shape=(5, 5), num_kernels=40, stride=5, padding=0,
        non_linear_func=lambda x: x)
print(cimg.shape)

cimg = lay.pool_layer(cimg, filter_size=(5, 5), stride=5, pooling_function=np.average)
print(cimg, cimg.shape)

cimg = lay.unravel_layer(cimg, 20)
print(cimg)

cimg = lay.MLP_layer(cimg, num_hidden_layers=5, hidden_size=10, activation_function=lambda x : x, output_layer_size=10, softmax_output=False)
print(cimg)

#utils.save_image('test_conv_lena512color.tiff', cimg)
'''

#Conv params
num_conv_layers = 4
kernel_shape_list = [(5,5), (3, 3) , (2, 2), (2, 2)]
num_kernels_list = [40, 30, 20, 20]
stride_list = [2, 3, 2, 1]
padding_list = [5, 5, 10, 5]
non_linearity_list = [lambda x : x] * 4
pooling_kernel_shape_list = [(5,5), (3, 3) , (2, 2), (2, 2)]
pooling_stride_list = [4, 3, 2, 1]
pooling_function_list = [np.average] * 4

#MLP Params
input_num_nodes = 40
num_hidden_layers=5
hidden_size=10
activation_function=lambda x : x
output_layer_size=10
softmax_output=False


cnn = CNN()
img_volume = np.random.random_sample((10, 100, 100))
out = cnn.forward(img_volume, num_conv_layers, kernel_shape_list, num_kernels_list,
            stride_list, padding_list, non_linearity_list, pooling_kernel_shape_list,
            pooling_stride_list, pooling_function_list, input_num_nodes, num_hidden_layers,
            hidden_size, activation_function, output_layer_size, softmax_output)
  
print(out)

