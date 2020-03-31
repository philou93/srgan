import numpy as np
import cv2


def pad_image(image, out_shape):
    container = np.zeros(out_shape)
    container[0:image.shape[0], 0:image.shape[1], :] = image
    return container


def resize_keep_ratio(image, new_size, interpolation):
    if image.shape[0] > image.shape[1]:
        biggest_size = 0
    else:
        biggest_size = 1

    ratio = image.shape[abs(biggest_size-1)] / image.shape[biggest_size]
    new_size[abs(biggest_size-1)] = int(new_size[abs(biggest_size-1)] * ratio)
    image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=interpolation)
    return image


def preprocess_imgs(image, hr_input_dims, lr_factor, pad=False, interpolation=cv2.INTER_CUBIC):
    x, y = hr_input_dims[:2]
    output_img = resize_keep_ratio(image, hr_input_dims[:2], interpolation)

    if pad is True:
        output_img = pad_image(output_img, (x, y, 3))

    image_input = resize_keep_ratio(output_img, [int(x / lr_factor), int(y / lr_factor)], interpolation)
    image_input = resize_keep_ratio(image_input, [x, y], interpolation)

    image_input = image_input / 255
    output_img = output_img / 255

    return image_input, output_img
