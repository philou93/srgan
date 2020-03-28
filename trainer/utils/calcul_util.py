import numpy as np


def pad_image(image, out_shape):
    container = np.zeros(out_shape)
    container[0:image.shape[0], 0:image.shape[1], :] = image
    return container
