import numpy as np
import utils

scale = 1


class MLP:
    def __init__(self, num_inputs, num_hidden_layers, hidden_size,
                 activation_function, activation_func_derv, output_layer_size, softmax_output=True,
                 optimization='Adam', lr=0.1, gamma=0.9, alpha=0.01, eps=1e-8, beta1=0.9, beta2=0.99):
        # Init params
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.activation_func_derv = activation_func_derv
        self.output_layer_size = output_layer_size
        self.softmax_output = softmax_output
        self.optimzation = optimization
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

        # Forward vectors
        self.Zhh = np.zeros((self.num_hidden_layers, 1, self.hidden_size))

        # Init weights
        self.Wxh = np.random.uniform(-scale, scale, (num_inputs, self.hidden_size)) * scale
        self.Whh = np.random.uniform(-scale, scale, (self.num_hidden_layers - 1, self.hidden_size, self.hidden_size)) * scale
        self.Why = np.random.uniform(-scale, scale, (self.hidden_size, self.output_layer_size)) * scale

        # Init Velocities
        self.Vwxh = np.zeros((num_inputs, self.hidden_size))
        self.Vwhh = np.zeros((self.num_hidden_layers - 1, self.hidden_size, self.hidden_size))
        self.Vwhy = np.zeros((self.hidden_size, self.output_layer_size))

        #Init linear cache
        self.Lwxh = np.zeros((num_inputs, self.hidden_size))
        self.Lwhh = np.zeros((self.num_hidden_layers - 1, self.hidden_size, self.hidden_size))
        self.Lwhy = np.zeros((self.hidden_size, self.output_layer_size))

        #Init Quad. cache
        self.Cwxh = np.zeros((num_inputs, self.hidden_size))
        self.Cwhh = np.zeros((self.num_hidden_layers - 1, self.hidden_size, self.hidden_size))
        self.Cwhy = np.zeros((self.hidden_size, self.output_layer_size))

        #Gradient matrices
        self.dWxh = np.zeros((num_inputs, self.hidden_size))
        self.dWhh = np.zeros((self.num_hidden_layers - 1, self.hidden_size, self.hidden_size))
        self.dWhy = np.zeros((self.hidden_size, self.output_layer_size))


    def forward(self, input_vector):
        input_vector = np.reshape(input_vector, (1, len(input_vector)))
        self.inp_vec = input_vector

        self.Zhh[0] = np.dot(input_vector, self.Wxh)
        inp_hin = self.activation_function(self.Zhh[0])


        for h in range(self.num_hidden_layers - 1):
            self.Zhh[h + 1] = np.dot(inp_hin, self.Whh[h])
            inp_hnext = self.activation_function(self.Zhh[h + 1])
            inp_hin = inp_hnext

        output_vector = self.activation_function(np.dot(inp_hin, self.Why))
        self.out_vec = output_vector
        output_vector = np.reshape(output_vector, (self.output_layer_size,))

        if self.softmax_output:
            output_vector = utils.softmax(output_vector)

        return output_vector

    def backward(self, delta_next):
        self.last_delta_next = delta_next
        delta = delta_next.reshape((1, delta_next.shape[0])) * self.activation_func_derv(self.out_vec)
        self.dWhy = np.dot(np.transpose(self.activation_function(self.Zhh[self.num_hidden_layers - 1])), delta)
        delta = np.dot(delta, np.transpose(self.Why)) * self.activation_func_derv(self.Zhh[self.num_hidden_layers - 1])

        for h in reversed(range(self.num_hidden_layers - 1)):
            self.dWhh[h] = np.dot(np.transpose(self.activation_function(self.Zhh[h])), delta)
            delta = np.dot(delta, np.transpose(self.dWhh[h])) * self.activation_func_derv(self.Zhh[h])

        self.dWxh = np.dot(np.transpose(self.inp_vec), delta)

    def step(self, t):
        if self.optimzation == 'Stochastic':
            self.stochastic_grad_step()
        elif self.optimzation == 'Momentum':
            self.momentum_grad_step()
        elif self.optimzation == 'Nesterov':
            self.nesterov_momentum_grad_step()
        elif self.optimzation == 'Adagrad':
            self.adagrad_step()
        elif self.optimzation == 'RMSProp':
            self.rmsprop_step()
        elif self.optimzation == 'Adam':
            self.adam_step(t)
        else:
            self.stochastic_grad_step()

    def stochastic_grad_step(self):
        self.Wxh -= self.lr * self.dWxh
        self.Whh -= self.lr * self.dWhh
        self.Why -= self.lr * self.dWhy

    def momentum_grad_step(self):
        self.Vwxh = self.gamma * self.Vwxh - self.alpha * self.dWxh
        self.Wxh += self.Vwxh

        self.Vwhh = self.gamma * self.Vwhh - self.alpha * self.dWhh
        self.Whh += self.Vwhh

        self.Vwhy = self.gamma * self.Vwhy - self.alpha * self.dWhy
        self.Why += self.Vwhy

    def nesterov_momentum_grad_step(self):
        tWxh = self.Wxh
        tWhh = self.Whh
        tWhy = self.Why

        # predicted point's grad
        self.Wxh += self.Vwxh * self.gamma
        self.Whh += self.Vwhh * self.gamma
        self.Why += self.Vwhy * self.gamma

        self.backward(self.last_delta_next)

        self.Wxh = tWxh
        self.Whh = tWhh
        self.Why = tWhy

        self.momentum_grad_step()

    def adagrad_step(self):
        self.Cwxh += np.square(self.dWxh)
        self.Wxh -= self.lr * self.dWxh / (np.sqrt(self.Cwxh) + self.eps)

        self.Cwhh += np.square(self.dWhh)
        self.Whh -= self.lr * self.dWhh / (np.sqrt(self.Cwhh) + self.eps)

        self.Cwhy += np.square(self.dWhy)
        self.Why -= self.lr * self.dWhy / (np.sqrt(self.Cwhy) + self.eps)

    def rmsprop_step(self):
        self.Cwxh = self.gamma * self.Cwxh + (1.0 - self.gamma) * np.square(self.dWxh)
        self.Wxh -= self.lr * self.dWxh / (np.sqrt(self.Cwxh) + self.eps)

        self.Cwhh += self.gamma * self.Cwhh + (1.0 - self.gamma) * np.square(self.dWhh)
        self.Whh -= self.lr * self.dWhh / (np.sqrt(self.Cwhh) + self.eps)

        self.Cwhy += self.gamma * self.Cwhy + (1.0 - self.gamma) * np.square(self.dWhy)
        self.Why -= self.lr * self.dWhy / (np.sqrt(self.Cwhy) + self.eps)

    def adam_step(self, t):
        self.Lwxh = self.beta1 * self.Lwxh + (1.0 - self.beta1) * self.dWxh
        self.Cwxh = self.beta2 * self.Cwxh + (1.0 - self.beta2) * np.square(self.dWxh)

        l_k_hat = self.Lwxh / (1. - self.beta1 ** t)
        c_k_hat = self.Cwxh / (1. - self.beta2 ** t)

        self.Wxh -= self.lr * l_k_hat / (np.sqrt(c_k_hat) + self.eps)

        self.Lwhh = self.beta1 * self.Lwhh + (1.0 - self.beta1) * self.dWhh
        self.Cwhh = self.beta2 * self.Cwhh + (1.0 - self.beta2) * np.square(self.dWhh)

        l_k_hat = self.Lwhh / (1. - self.beta1 ** t)
        c_k_hat = self.Cwhh / (1. - self.beta2 ** t)

        self.Whh -= self.lr * l_k_hat / (np.sqrt(c_k_hat) + self.eps)

        self.Lwhy = self.beta1 * self.Lwhy + (1.0 - self.beta1) * self.dWhy
        self.Cwhy = self.beta2 * self.Cwhy + (1.0 - self.beta2) * np.square(self.dWhy)

        l_k_hat = self.Lwhy / (1. - self.beta1 ** t)
        c_k_hat = self.Cwhy / (1. - self.beta2 ** t)

        self.Why -= self.lr * l_k_hat / (np.sqrt(c_k_hat) + self.eps)
