import os

import cv2
import numpy as np

from tester.enhance_image.argsparser import parse_args
from trainer.models.generator import Generator


def convert_to_rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


def get_images(image_paths, new_size):
    inputs = []
    for image_path in image_paths:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        try:
            new_size = int(new_size)
            new_size = (original_image.shape[1] * new_size, original_image.shape[0] * new_size)
        except:
            new_size = new_size.split(",")
        finally:
            if len(new_size) != 2:
                raise ValueError("'new-size' doit être un facteur (ex: 2) ou une dimention (ex: 800,850)")
            try:
                new_size = tuple([int(i) for i in new_size])
            except:
                raise ValueError("Les valeurs de 'new-size' doivent être des entiers.")
        upscale_original_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_CUBIC)
        upscale_original_image = upscale_original_image / 255
        inputs.append(upscale_original_image)

    return inputs


def main(args):
    inputs = get_images(args.images, args.new_size)
    generator = Generator()  # IMPORTANT: Il faut recréer le modèle exacte à l'entrainement
    generator.load_weights(args.gen_path)

    for image, path in list(zip(inputs, args.images)):
        # original_size = original.shape
        generate_output = generator.forward(np.array([image]))
        generate_img = np.array(np.clip(generate_output[0] * 255, 0, 255), dtype=np.int)
        path = str.replace(path, '\\', '/')
        new_name = f"enhanced_{path.split('/')[-1]}"
        new_path = os.path.join(args.save_to, new_name)
        cv2.imwrite(new_path, generate_img)
        print(f"Image enregistrée à {new_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
