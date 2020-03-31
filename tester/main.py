import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tester.argsparser import parse_args

from trainer.utils.calcul_util import preprocess_imgs
from trainer.generator import Generator

hr_input_dims = [200, 200, 3]


def crop(image, shape):
    return image[:shape[0], :shape[1], :]


def get_images(image_paths, lr_factor):
    dataset = []
    originals = []
    for image_path in image_paths:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_input, output_img = preprocess_imgs(original_image, hr_input_dims, lr_factor, pad=False)
        dataset.append(image_input)
        originals.append(output_img)

    return dataset, originals


def show_result(original_img, input_img, output_img, lr_factor=4):
    plt.title("Image originale")
    plt.imshow(original_img)
    plt.show()
    plt.title(f"Image réduite ({lr_factor}x)")
    plt.imshow(input_img)
    plt.show()
    plt.title("Résultat")
    plt.imshow(output_img, )
    plt.show()


def main(args):
    inputs, originals = get_images(args.images, args.factor)
    generator = Generator((None,None,3), (None, None, 3), nb_filter_conv1=16)
    generator.load_weights(args.gen_path)

    for input_img, original in list(zip(inputs, originals)):
        # original_size = original.shape
        generate_output = generator.forward(np.array([input_img]))
        generate_img = np.array(generate_output[0] * 255, dtype=np.int)
        input_img = np.array(input_img * 255, dtype=np.int)
        show_result(original, input_img, generate_img, lr_factor=args.factor)


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gcloud.json"
    args = parse_args()
    main(args)
