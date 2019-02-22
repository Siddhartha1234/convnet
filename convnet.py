from layers import Layers

class CNN:
    def convlayer_composition(self, input_image_volume, num_conv_layers,
            kernel_shape_list, stride_list, padding_list, non_linearity_list,
            pooling_kernel_shape_list, pooling_stride_list, pooling_function_list):
        inp = input_image_volume
        for l in range(num_conv_layers):
            inp = Layers.conv_layer(inp, kernel_shape_list[l],
                    stride_list[l], padding_list[l], non_linearity_list[l])
            inp = Layers.pool_layer(inp, pooling_kernel_shape_list[l],
                    pooling_stride_list[l], pooling_function_list[l])

        return inp

    def forward(self, input_image_volume, num_conv_layers,
            kernel_shape_list, stride_list, padding_list, non_linearity_list,
            pooling_kernel_shape_list, pooling_stride_list,
            pooling_function_list, mlp_num_hidden_layers, mlp_hidden_size,
            mlp_activation_function, final_output_size, final_softmax=True):
        inp = self.convlayer_composition(input_image_volume, num_conv_layers,
            kernel_shape_list, stride_list, padding_list, non_linearity_list,
            pooling_kernel_shape_list, pooling_stride_list,
            pooling_function_list)

        out_vector = Layers.MLP_layer(inp, mlp_num_hidden_layers,
                mlp_hidden_size, mlp_activation_function, final_output_size,
                final_softmax)

        return out_vector
