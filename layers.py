from conv import Conv
import numpy as np
import utils

scale = 0.01

class ConvLayer:
    def __init__(self, kernel_shape, num_kernels, stride, padding, non_linear_func, prev_nfeatures):
        # Init params
        self.convc = Conv()
        self.kernel_shape = kernel_shape
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding
        self.non_linear_func = non_linear_func
        
        # Init weights

        self.kernel = np.random.uniform(0, +scale, (num_kernels, prev_nfeatures, kernel_shape[0], kernel_shape[1]))


    def forward(self, input_vector_volume):
        self.inp_vec = input_vector_volume
        if len(self.inp_vec.shape) == 2:
            self.inp_vec = np.expand_dims(self.inp_vec, axis=0)

        output_array = []
        for n in range(self.num_kernels):
            output_array.append(self.convc.conv(input_vector_volume,
                self.kernel[n], self.stride, self.padding, self.non_linear_func))
        #print("Shape after conv layer",  np.array(output_array).shape)
        self.out_vec = np.array(output_array)
        return self.out_vec

    def backward(self, delta_nxt, nxt_layer):
        delta = np.zeros(self.out_vec.shape)
        indices_vec = nxt_layer.indices_vec

        for n in range(nxt_layer.num_kernels):
            for i in range(0, self.out_vec.shape[1] - nxt_layer.filter_size[0] + 1, nxt_layer.stride):
                for j in range(0,  self.out_vec.shape[2] - nxt_layer.filter_size[1] + 1, nxt_layer.stride):
                    delta[n][i: i + nxt_layer.filter_size[0], j: j + nxt_layer.filter_size[1]] = indices_vec[n][i: i + nxt_layer.filter_size[0], j: j + nxt_layer.filter_size[1]] * delta_nxt[n][i // nxt_layer.stride, j // nxt_layer.stride]
        self.dkernel = []
        for n in range(self.num_kernels):
            temp_arr = []
            for x in range(self.inp_vec.shape[0]):
                temp_arr.append(self.convc.conv(self.inp_vec[x], delta[n], 1, 0, utils.relu))
            self.dkernel.append(temp_arr)
        self.dkernel = np.array(self.dkernel)
        # print("kernel shapes", self.kernel.shape, self.dkernel.shape)
        return delta

    def grad_step(self, lr):
        # print(self.dkernel)
        self.kernel -= lr * self.dkernel


class PoolLayer:
    def __init__(self, filter_size, num_kernels, stride, pooling_function):
        self.convc = Conv()
        self.filter_size = filter_size
        self.stride = stride
        self.num_kernels = num_kernels
        self.pooling_function = pooling_function
        
    def forward(self, input_vector_volume):
        output_array = []
        indices_array = []
        for i in range(input_vector_volume.shape[0]):
            out, indices = self.convc.pool(input_vector_volume[i],
                                                self.filter_size, self.stride, self.pooling_function)
            output_array.append(out)
            indices_array.append(indices)
        #print("Shape after pool layer", np.array(output_array).shape)

        self.out_vec = np.array(output_array)
        self.indices_vec = np.array(indices_array)
        return self.out_vec
    
    def backward(self, delta_next, nxt_lay):
        if nxt_lay.__class__.__name__ == "DenseLayer":
            delta = np.reshape(delta_next, self.out_vec.shape)
        elif nxt_lay.__class__.__name__ == "ConvLayer":
            K = np.swapaxes(nxt_lay.kernel, 0, 1)
            delta = []
            for n in range(self.num_kernels):
                delta.append(self.convc.conv(delta_next, K[n], 1, K.shape[-1] - 1, utils.relu))
            delta = np.array(delta)
        return delta

    def grad_step(self, lr):
        pass

class DenseLayer:
    def __init__(self, num_inputs, output_layer_size, activation_function, softmax_output=True):
        # Init params
        self.num_inputs = num_inputs
        self.output_layer_size = output_layer_size
        self.activation_function = activation_function
        self.softmax_output = softmax_output

        # Init weights
        self.Wxy = np.random.uniform(0, +scale, (num_inputs, output_layer_size))
        self.dWxy = np.zeros((num_inputs, output_layer_size))

    def forward(self, input_vector):
        self.input_vector = self.activation_function(np.reshape(input_vector.flatten(), (1, input_vector.flatten().shape[0])))
        output_vector = np.dot(self.input_vector, self.Wxy)
        output_vector = np.reshape(output_vector, (self.output_layer_size,))

        if self.softmax_output:
            output_vector = utils.softmax(output_vector)

        return output_vector

    def backward(self, delta_next, nxt_lay):
        delta = delta_next.reshape((1, delta_next.shape[0]))
        self.dWxy = np.dot(np.transpose(self.input_vector), delta)
        return np.dot(delta, np.transpose(self.Wxy))

    def grad_step(self, lr):
       # print(self.dWxy)
        self.Wxy -= lr * self.dWxy


class UnravelLayer:
    def __init__(self, output_num_nodes):
        self.output_num_nodes = output_num_nodes
        self.weight_unravel = None

    def forward(self, input_vector_volume):
        input_vec = input_vector_volume.flatten()
        raveled_input_vector_volume = np.reshape(input_vec,
                                                 (1, input_vec.shape[0]))
        if self.weight_unravel is None:
            self.weight_unravel = np.random.random_sample((raveled_input_vector_volume.shape[1],
                                                           self.output_num_nodes))

        return np.dot(raveled_input_vector_volume, self.weight_unravel)

    def backward(self):
        pass

    def grad_step(self, lr):
        pass


class MLPLayer:
    def __init__(self, num_inputs, num_hidden_layers, hidden_size,
                 activation_function, output_layer_size, softmax_output=True):
        # Init params
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.output_layer_size = output_layer_size
        self.softmax_output = softmax_output
        
        # Init weights
        self.Wxh = np.random.random_sample((num_inputs, self.hidden_size))
        self.Whh = np.random.random_sample((self.num_hidden_layers - 1, self.hidden_size, self.hidden_size))
        self.Why = np.random.random_sample((self.hidden_size, self.output_layer_size))

    def forward(self, input_vector):
        inp_hin = self.activation_function(np.dot(input_vector, self.Wxh))

        for h in range(self.num_hidden_layers - 1):        
            inp_hnext = self.activation_function(np.dot(inp_hin, self.Whh[h]))
            inp_hin = inp_hnext
            
        output_vector = self.activation_function(np.dot(inp_hin, self.Why))
        output_vector = np.reshape(output_vector, (self.output_layer_size,))

        if self.softmax_output:
            output_vector = utils.softmax(output_vector)

        return output_vector
    
    def backward(self, delta_next):
        pass

    def grad_step(self, lr):
        pass
