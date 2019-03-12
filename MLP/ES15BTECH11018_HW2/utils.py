import csv
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigm_derv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - max(x)
    val = np.exp(x) / np.sum(np.exp(x))
    return val


def cross_entropy_loss(out_vector, target, eps=1e-12):
    out_vector = np.clip(out_vector, eps, 1 - eps)
    return -np.sum(target * np.log(out_vector + eps))  # Assume cross entropy loss and softmax on output




