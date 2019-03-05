import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from convnet import CNN
import utils
import matplotlib.pyplot as plt


class MNIST:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.get_mnist_data()
        self.init_model()

    def get_mnist_data(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.X_train = np.array([img.reshape((28, 28)) for img in mnist.train.images])
        self.y_train = mnist.train.labels
        self.X_test = np.array([img.reshape((28, 28)) for img in mnist.test.images])
        self.y_test = mnist.test.labels
        del mnist

    def init_model(self):
        # Conv params
        self.num_channels = 1
        self.num_conv_layers = 2
        self.kernel_shape_list = [(5, 5), (5, 5)]
        self.num_kernels_list = [32, 64]
        self.stride_list = [1, 1]
        self.padding_list = [0, 0]
        self.non_linearity_list = [utils.relu, utils.relu]
        self.pooling_kernel_shape_list = [(2, 2), (2, 2)]
        self.pooling_stride_list = [2, 2]
        self.pooling_function_list = [np.max] * 2

        # Dense Params
        self.input_num_nodes = 1024
        self.output_layer_size = 10
        self.softmax_output = True

        # Opt params
        self.learning_rate = 0.1
        self.epochs = 15

        self.model = CNN(self.num_channels, self.num_conv_layers, self.kernel_shape_list, self.num_kernels_list,
                          self.stride_list, self.padding_list, self.non_linearity_list, self.pooling_kernel_shape_list,
                          self.pooling_stride_list, self.pooling_function_list, self.input_num_nodes, self.output_layer_size, self.softmax_output,
                          self.learning_rate)

    def train(self, batch_size=1): #Default is stochastic gradient descent
        self.training_loss_array = []
        self.test_loss_array = []
        for n_epoch in range(self.epochs):
            for i in range(0, self.X_train.shape[0], batch_size):
                dloss = np.zeros((10,))
                batch_loss = 0.0
                for b in range(batch_size):
                    out = self.model.forward(self.X_train[i + b])
                    # print(out)
                    tar = self.y_train[i + b]
                    dloss += out - tar
                    batch_loss += utils.cross_entropy_loss(out, tar)
                dloss /= batch_size
                batch_loss /= batch_size
                print("Epoch :", n_epoch, "ind:", i, "batch loss:", batch_loss)
                self.model.backward(dloss)
            # Find avg. training loss
            training_loss = 0.0
            for i in range(self.X_train.shape[0]):
                out = self.model.forward(self.X_train[i], sample=True)
                tar = self.y_train[i]
                training_loss += utils.cross_entropy_loss(out, tar)
            training_loss /= self.X_train.shape[0]
            print("Training loss after epoch", n_epoch, "is :", training_loss)
            self.training_loss_array.append(training_loss)
            # Find avg. test loss
            test_loss = 0.0
            accuracy = 0.0
            for i in range(self.X_test.shape[0]):
                out = self.model.forward(self.X_test[i])
                tar = self.y_test[i]
                if tar[np.argmax(out)] == 1:
                    accuracy += 1
                test_loss += utils.cross_entropy_loss(out, tar)
            test_loss /= self.X_test.shape[0]
            accuracy /= self.X_test.shape[0]
            print("Test loss after epoch", n_epoch, "is :", test_loss)
            print("Accuracy after epoch", n_epoch, "is :", accuracy)

            plt.show()
            self.test_loss_array.append(test_loss)
            # Shuffle training data after every epoch
            p = np.random.permutation(self.X_train.shape[0])
            self.X_train = self.X_train[p]
            self.y_train = self.y_train[p]

    def visualize(self):
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.plot(range(self.epochs), self.training_loss_array, label='training loss')
        # plt.plot(range(self.epochs), self.test_loss_array, label='test loss')
        # plt.legend(loc='upper right')
        # plt.show()
        # plt.clf()
        self.model.activation_map_samples = np.array(self.model.activation_map_samples)
        fig, axs = plt.subplots(self.epochs, 4)

        for i in range(self.epochs):
            axs[i][0].imshow(utils.normalized_to_grayscale(self.model.activation_map_samples[i][0]))
            axs[i][0].set(xlabel='Conv1, epoch:' + str(i + 1))
            axs[i][1].imshow(utils.normalized_to_grayscale(self.model.activation_map_samples[i][1]))
            axs[i][1].set(xlabel='Pool1, epoch:' + str(i + 1))
            axs[i][2].imshow(utils.normalized_to_grayscale(self.model.activation_map_samples[i][2]))
            axs[i][2].set(xlabel='Conv2, epoch:' + str(i + 1))
            axs[i][3].imshow(utils.normalized_to_grayscale(self.model.activation_map_samples[i][3]))
            axs[i][3].set(xlabel='Pool2, epoch:' + str(i + 1))
        plt.show()

if __name__ == '__main__':
    mn = MNIST()
    mn.get_mnist_data()
    mn.train(batch_size=10)
    mn.visualize()