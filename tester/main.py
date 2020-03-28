import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tester.utils.argsparser import parse_args

from trainer.generator import Generator
from trainer.utils.calcul_util import downsize_dims_by_factor_no_channel, downsize_dims_by_factor
from trainer.utils.normalize_fcts import normalize_between_0_1, unnormalize_between_0_1

hr_input_dims = [320, 640, 3]


def get_images(image_paths):
    dataset = []

    if hr_input_dims[-1] == 1:
        read_flag = cv2.IMREAD_GRAYSCALE
    else:
        read_flag = cv2.IMREAD_COLOR

    for path in image_paths:
        image = cv2.imread(path, read_flag)
        image = np.atleast_3d(image)
        dataset.append(image)

    return dataset


def downsize_images(images, lr_factor):
    dataset = []
    for image in images:
        if image.shape[0] < hr_input_dims[0] or image.shape[1] < hr_input_dims[1]:
            new_image = np.zeros(hr_input_dims)
            new_image[:min([image.shape[0], hr_input_dims[0]]), :min([image.shape[1], hr_input_dims[1]]), :] = \
                image[:min([image.shape[0], hr_input_dims[0]]), :min([image.shape[1], hr_input_dims[1]]), :]
        else:
            new_image = image[:hr_input_dims[0], :hr_input_dims[1], :]

        downsampling_img = cv2.resize(new_image, downsize_dims_by_factor_no_channel(hr_input_dims, lr_factor)[::-1],
                                      interpolation=cv2.INTER_NEAREST)
        downsampling_img = normalize_between_0_1(np.atleast_3d(downsampling_img))

        dataset.append([downsampling_img, new_image])

    return dataset


def show_result(original_img, input_img, output_img, lr_factor=4):
    plt.title("Image originale")
    plt.imshow(original_img)
    plt.show()
    plt.title(f"Image réduite ({lr_factor}x)")
    plt.imshow(input_img)
    plt.show()
    plt.title("Résultat")
    plt.imshow(output_img)
    plt.show()


def main(args):
    images = get_images(args.images)
    images = downsize_images(images, args.factor)
    generator = Generator(downsize_dims_by_factor(hr_input_dims, args.factor), hr_input_dims, nb_filter_conv1=16)
    generator.load_weights(args.gen_path)

    for downsize_img, original_img in images:
        generate_output = generator.forward(np.array([downsize_img]))
        generate_img = unnormalize_between_0_1(generate_output)
        input_img = unnormalize_between_0_1(downsize_img)
        show_result(original_img, input_img, generate_img, lr_factor=args.factor)


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gcloud.json"
    args = parse_args()
    main(args)
