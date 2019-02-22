from PIL import Image
import numpy as np

class utils:
    def read_image(path):
        img = Image.open(path)
        return np.array(img)

    def save_image(path, img):
        img = Image.fromarray(img)
        img.save(path)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
