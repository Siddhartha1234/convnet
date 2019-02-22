from PIL import Image
import numpy as np

class utils:
    def read_image(path):
        img = Image.open(path)
        return np.array(img)

    def save_image(path, img):
        img = Image.fromarray(img)
        img.save(path)
