import numpy as np


class Conv:
    def conv(self, input_vector, kernel_shape, stride, padding,
             non_linear_func):
        if len(input_vector.shape) == 2:
            input_vector = np.expand_dims(input_vector, axis=0)

        filter_kernel = np.random.random_sample((input_vector.shape[0], kernel_shape[0], kernel_shape[1]))
        inp_vec_with_padding = np.pad(input_vector, ((0, 0), (padding, padding),
                                                     (padding, padding)), 'constant', constant_values=0)
        output_vector = np.zeros(((inp_vec_with_padding.shape[1] -
                                   kernel_shape[0]) // stride + 1,
                                  (inp_vec_with_padding.shape[2] -
                                   kernel_shape[1]) // stride + 1))
        # print(output_vector.shape)
        for c in range(input_vector.shape[0]):
            for i in range(0, inp_vec_with_padding.shape[1] -
                              kernel_shape[0] + 1, stride):
                for j in range(0, inp_vec_with_padding.shape[2] -
                                  kernel_shape[1] + 1, stride):
                    for s in range(0, kernel_shape[0]):
                        for t in range(0, kernel_shape[1]):
                            output_vector[i // stride][j // stride] += inp_vec_with_padding[c][i + s][j + t] * \
                                                                       filter_kernel[c][s][t]

                    output_vector[i // stride][j // stride] = non_linear_func(output_vector[i // stride][j // stride])
        return output_vector

    def pool(self, input_vector, filter_size, stride, pooling_function):
        output_vector = np.zeros(((input_vector.shape[0] - filter_size[0]) //
                                  stride + 1, (input_vector.shape[1] - filter_size[1]) // stride + 1))

        for i in range(0, input_vector.shape[0] - filter_size[0] + 1, stride):
            for j in range(0, input_vector.shape[1] - filter_size[1] + 1, stride):
                output_vector[i // stride][j // stride] = pooling_function(
                    input_vector[i: i + filter_size[0], j: j + filter_size[1]])

        return output_vector
