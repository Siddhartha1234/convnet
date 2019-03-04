import numpy as np

class Conv:
    def conv(self, input_vector, filter_kernel, stride, padding,
             non_linear_func):
        if len(input_vector.shape) == 2:
            input_vector = np.expand_dims(input_vector, axis=0)
        if len(filter_kernel.shape) == 2:
            filter_kernel = np.expand_dims(filter_kernel, axis=0)

        #Filter kernel flip
        np.rot90(filter_kernel, 2, axes=(1, 2))

        kernel_shape = filter_kernel.shape[1], filter_kernel.shape[2]
        inp_vec_with_padding = np.pad(input_vector, ((0, 0), (padding, padding),
                                                     (padding, padding)), 'constant', constant_values=0)

        output_vector = np.zeros(((inp_vec_with_padding.shape[1] -
                                   kernel_shape[0]) // stride + 1,
                                  (inp_vec_with_padding.shape[2] -
                                   kernel_shape[1]) // stride + 1))
        for c in range(input_vector.shape[0]):
            for i in range(0, inp_vec_with_padding.shape[1] -
                              kernel_shape[0] + 1, stride):
                for j in range(0, inp_vec_with_padding.shape[2] -
                                  kernel_shape[1] + 1, stride):
                    output_vector[i // stride][j // stride] += np.sum(np.multiply(inp_vec_with_padding[c][i : i + kernel_shape[0], j : j + kernel_shape[1]], filter_kernel[c]))
                    output_vector[i // stride][j // stride] = non_linear_func(output_vector[i // stride][j // stride])
        return output_vector

    def pool(self, input_vector, filter_size, stride, pooling_function):
        output_vector = np.zeros(((input_vector.shape[0] - filter_size[0]) //
                                  stride + 1, (input_vector.shape[1] - filter_size[1]) // stride + 1))
        if pooling_function == np.max:
            indices_vector = np.zeros(input_vector.shape)
        else:
            indices_vector = np.ones(input_vector.shape)

        for i in range(0, input_vector.shape[0] - filter_size[0] + 1, stride):
            for j in range(0, input_vector.shape[1] - filter_size[1] + 1, stride):
                output_vector[i // stride][j // stride] = pooling_function(
                    input_vector[i: i + filter_size[0], j: j + filter_size[1]])
                if pooling_function == np.max:
                    temp_vector = input_vector[i: i + filter_size[0], j: j + filter_size[1]]
                    ind, jind = np.unravel_index(np.argmax(temp_vector, axis=None), temp_vector.shape)
                    indices_vector[i + ind, j + jind] = 1
                indices_vector[i: i + filter_size[0],j : j + filter_size[1]] /= np.sum(indices_vector[i: i + filter_size[0],j : j + filter_size[1]])
        return output_vector, indices_vector
