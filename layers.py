from conv import Conv
import numpy as np
from utils import utils

class Layers:
    def __init__(self):
        self.convc = Conv()
    def conv_layer(input_vector_volume, kernel_shape, stride, padding,
            non_linear_func):
        output_array = []
        for i in range(input_vector_volume.shape[0]):
            output_array.append(self.convc.conv(input_vector_volume[i],
                kernel_shape[0], stride, padding, non_linear_func))

        return np.array(output_array)

    def pool_layer(input_vector_volume, filter_size, stride, pooling_function):
        output_array = []
        for i in range(input_vector_volume.shape[0]):
            output_array.append(self.convc.pool(input_vector_volume[i],
                filter_size[0], stride, pooling_function)
        return np.array(output_array)

    def unravel_layer(input_vector_volume, output_num_nodes):
        raveled_input_vector_volume = np.reshape(input_vector_volume.flatten(),
            (1, input_vector_volume.shape[0]))

        weight_unravel = np.random.random_sample((raveled_input_vector_volume.shape[1],
            output_num_nodes))

        return np.dot(raveled_input_vector_volume, weight_unravel)

    def MLP_layer(input_vector, num_hidden_layers, hidden_size,
        activation_function, output_layer_size, softmax_output=True):

        Wxh = np.random.random_sample((input_vector.shape[1], hidden_size))

        inp_hin = activation_function(np.dot(input_vector, Wxh))

        for h in range(num_hidden_layers - 1):
            Whh = np.random.random_sample((hidden_size, hidden_size))
            inp_hnext = activation_function(np.dot(inp_hin, Whh))
            inp_hin = inp_hnext

        Why = np.random.random_sample((hidden_size, output_layer_size))

        output_vector = activation_function(np.dot(inp_hin, Why))

        if softmax_output:
            output_vector = utils.softmax(output)

        return output_vector
