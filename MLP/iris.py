import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

from mlp import MLP
from optim import Stochastic, Momentum, Nesterov, Adagrad, RMSProp, Adam

import utils

class Iris:
    def __init__(self, epochs=500):
        self.get_iris_data()
        self.model = MLP(num_inputs=4, num_hidden_layers=3, hidden_size=5, activation_function=utils.sigmoid,
                         activation_func_derv=utils.sigm_derv, output_layer_size=3, softmax_output=True,
                         optimization='Stochastic')
        self.epochs = epochs

    def get_iris_data(self):
        with open('iris-data.csv') as f:
            reader = csv.reader(f, delimiter=',')
            X = []
            Y = []
            ref = {}
            cnt = -1
            for row in reader:
                X.append(row[:-1])
                if row[-1] not in ref:
                    cnt += 1
                    ref[row[-1]] = cnt
                Y.append(cnt)
            X = np.array(X)
            Y = np.array(Y)

            X_train = []
            Y_train = []

            X_test = []
            Y_test = []
            for cnt in range(3):
                XX = X[np.where(Y == cnt)].tolist()
                YY = Y[np.where(Y == cnt)].tolist()
                X_train.extend(XX[:30])
                X_test.extend(XX[30:])
                Y_train.extend(YY[:30])
                Y_test.extend(YY[30:])
            YY_train = []
            YY_test = []
            for y in range(len(Y_train)):
                x = [0, 0, 0]
                x[Y_train[y]] = 1
                YY_train.append(x)
            for y in range(len(Y_test)):
                x = [0, 0, 0]
                x[Y_test[y]] = 1
                YY_test.append(x)

            self.X_train = np.array(X_train, dtype="float32")
            self.X_test = np.array(X_test, dtype="float32")
            self.y_train = np.array(YY_train, dtype="float32")
            self.y_test = np.array(YY_test, dtype="float32")

            print(self.X_train, self.y_train)
            print(self.X_test, self.y_test)

    def train_and_evaluate(self, batch_size=1):  # Default is stochastic gradient descent
        self.training_loss_array = []
        self.test_loss_array = []
        self.test_accuracy_array = []

        for n_epoch in range(self.epochs):
            for i in range(0, self.X_train.shape[0], batch_size):
                dloss = np.zeros((3,))
                batch_loss = 0.0
                for b in range(batch_size):
                    out = self.model.forward(self.X_train[i])
                    tar = self.y_train[i]
                    dloss += out - tar
                    batch_loss += utils.cross_entropy_loss(out, tar)
                dloss /= batch_size
                batch_loss /= batch_size
                self.model.backward(dloss)
                self.model.step(n_epoch)

            # Find avg. training loss
            training_loss = 0.0
            for i in range(self.X_train.shape[0]):
                out = self.model.forward(self.X_train[i])
                tar = self.y_train[i]
                training_loss += utils.cross_entropy_loss(out, tar)
            training_loss /= self.X_train.shape[0]
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

            print("Epoch :", n_epoch, "Train error", training_loss, "test loss", test_loss, "Accuracy", accuracy)
            self.test_loss_array.append(test_loss)
            self.test_accuracy_array.append(accuracy)

            # Shuffle training data after every epoch
            p = np.random.permutation(self.X_train.shape[0])
            self.X_train = self.X_train[p]
            self.y_train = self.y_train[p]

    def visualize(self):
        plt.xlabel('epoch')
        plt.ylabel('Stochastic Train/Test loss')
        plt.plot(range(self.epochs), self.training_loss_array, label='training loss')
        plt.plot(range(self.epochs), self.test_loss_array, label='test loss')
        #plt.plot(range(self.epochs), self.test_accuracy_array, label='Accuracy on test set')
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    ir = Iris(epochs=5000)
    ir.train_and_evaluate(batch_size=1)
    ir.visualize()
