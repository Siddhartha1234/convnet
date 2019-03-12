import numpy as np

class Stochastic:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.Wxh -= self.lr * self.model.dWxh
        self.model.Whh -= self.lr * self.model.dWhh
        self.model.Why -= self.lr * self.model.dWhy

class Momentum:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.grad_step(self.lr)

class Nesterov:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.grad_step(self.lr)

class Adagrad:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.grad_step(self.lr)

class RMSProp:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.grad_step(self.lr)

class Adam:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.grad_step(self.lr)

