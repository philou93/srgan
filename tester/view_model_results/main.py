import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tester.view_model_results.argsparser import parse_args
from trainer.models.generator import Generator
from trainer.utils.calcul_util import preprocess_imgs

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


def generate_heatmap(original_img, output_img):
    heatmap = original_img - output_img  # Permet de visualiser où sont les différences dans l'image (mae).
    heatmap = np.mean(heatmap, axis=2)
    plt.title("heatmap")
    plt.imshow(heatmap)
    plt.show()


def main(args):
    inputs, originals = get_images(args.images, args.factor)
    generator = Generator(nb_filter_conv1=32)  # IMPORTANT: Il faut recréer le modèle exacte à l'entrainement
    generator.load_weights(args.gen_path)

    for input_img, original in list(zip([inputs[0]], [originals[0]])):
        # original_size = original.shape
        generate_output = generator.forward(np.array([input_img]))
        generate_img = np.array(generate_output[0] * 255, dtype=np.int)
        input_img = np.array(input_img * 255, dtype=np.int)
        original = np.array(original * 255, dtype=np.int)
        show_result(original, input_img, generate_img, lr_factor=args.factor)
        generate_heatmap(original, generate_img)


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gcloud.json"
    args = parse_args()
    main(args)
