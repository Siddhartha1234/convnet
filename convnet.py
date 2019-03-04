from layers import ConvLayer, DenseLayer, PoolLayer
import numpy as np
import utils

class CNN:
    def __init__(self, num_channels, num_conv_layers, kernel_shape_list, num_kernels_list, stride_list, padding_list, non_linearity_list,
                 pooling_kernel_shape_list, pooling_stride_list,
                 pooling_function_list, dense_input_num_nodes, final_output_size, final_softmax=True,
                 learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layers = []
        for l in range(num_conv_layers):
            if l == 0:
                self.layers.append(ConvLayer(kernel_shape_list[l], num_kernels_list[l], stride_list[l], padding_list[l],
                                             non_linearity_list[l], num_channels))
            else:
                self.layers.append(ConvLayer(kernel_shape_list[l], num_kernels_list[l], stride_list[l], padding_list[l],
                                             non_linearity_list[l], num_kernels_list[l - 1]))
            self.layers.append(PoolLayer(pooling_kernel_shape_list[l], num_kernels_list[l], pooling_stride_list[l], pooling_function_list[l]))

        self.layers.append(DenseLayer(dense_input_num_nodes, final_output_size, utils.relu, final_softmax))

    def forward(self, input_image_volume):
        out_vector = input_image_volume
        for layer in self.layers:
            out_vector = layer.forward(out_vector)
        return out_vector


    def backward(self, delta_out):
        delta_nxt = delta_out
        for ind, layer in enumerate(self.layers[::-1]):
            nxt_lay = None
            if ind > 0:
                nxt_lay = self.layers[-ind]
            delta_nxt = layer.backward(delta_nxt, nxt_lay)
        for layer in self.layers:
            layer.grad_step(self.learning_rate)
