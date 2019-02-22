import numpy as np

class Conv:
    def conv(input_vector, kernel_shape, stride, padding,
            non_linear_func):
        filter_kernel = np.random.random_sample(kernel_shape)
        if len(input_vector.shape) == 2:
            input_vector = np.expand_dims(input_vector, axis=2)
            filter_kernel = np.expand_dims(filter_kernel, axis=2)
        inp_vec_with_padding = np.pad(input_vector, ((padding, padding),
            (padding, padding), (0,0)), 'constant', constant_values=0)
        #print(inp_vec_with_padding.shape)
        output_vector = np.zeros(((inp_vec_with_padding.shape[0] -
            filter_kernel.shape[0])//stride + 1,
            (inp_vec_with_padding.shape[1] -
                filter_kernel.shape[1])//stride + 1))
        #print(output_vector.shape)
        for i in range(0, inp_vec_with_padding.shape[0] -
                kernel_shape[0] + 1, stride):
            for j in range(0, inp_vec_with_padding.shape[1] -
                    kernel_shape[1] + 1, stride):
                for c in range(kernel_shape[2]):
                    for s in range(0, kernel_shape[0]):
                        for t in range(0, kernel_shape[1]):
                            output_vector[i // stride][j // stride] += inp_vec_with_padding[i + s][j + t][c] * filter_kernel[s][t][c]

                output_vector[i // stride][j // stride] = non_linear_func(output_vector[i // stride][j // stride])

        return output_vector

    def pool(input_vector, filter_size, stride, pooling_function):
        output_vector = np.zeros(((input_vector.shape[0] - filter_size[0]) //
            stride + 1 , (input_vector.shape[1]  - filter_size[1]) // stride + 1))

        for i in range(0, input_vector.shape[0] - filter_size[0] + 1, stride):
            for j in range(0, input_vector.shape[1] - filter_size[1] + 1, stride):
                output_vector[i // stride][j // stride] = pooling_function(
                        input_vector[i : i + filter_size[0], j : j + filter_size[1]])

        return output_vector
