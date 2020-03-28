import os
import random
from functools import partial

import cv2
import numpy as np

import trainer.config as config
from trainer.discriminator import Discriminator
from trainer.generator import Generator
from trainer.utils.argsparser import parse_args
from trainer.utils.calcul_util import pad_image
from trainer.utils.download import split_bucket, download_images_async

lr_factor = 4
hr_input_dims = [800, 800, 3]

assert hr_input_dims[2] in [3], "hr_input_dims[2] doit etre soit 1 ou 3"

index_img_to_show = [3, 8, 23]  # Hardcoder pour facilite la chose


def import_dataset(data_path, extension_file):
    global hr_input_dims
    if not isinstance(extension_file, tuple):
        extension_file = tuple(extension_file)

    dataset = []  # Va être des string si local et des blobs si sur gcloud.

    if not data_path.startswith("gs://"):
        image_paths = os.listdir(data_path)
        for path in image_paths:
            if path.endswith(extension_file):
                dataset.append(path)
    else:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket_name, sub_folder = split_bucket(data_path)
        blobs = storage_client.list_blobs(bucket_name, prefix=sub_folder)
        for blob in blobs:
            if blob.name.endswith(extension_file):
                dataset.append(blob)
    return dataset


def preprocessing(blobs, lr_factor, extension=None):
    images = download_images_async(blobs, extension_file=extension)
    # TODO: il doit bien y avoir une facon de remplacer le Fully-Connected par une Convolution.
    x, y = hr_input_dims[:2]

    train_x = []
    train_y = []
    for image in images:
        padded_image = pad_image(image, (x, y, 3))
        image_input = cv2.resize(padded_image,
                                 (int(x / lr_factor), int(y / lr_factor)),
                                 interpolation=cv2.INTER_CUBIC)
        image_input = cv2.resize(image_input,
                                 (x, y),
                                 interpolation=cv2.INTER_CUBIC)
        train_x.append(image_input / 255)
        train_y.append(padded_image / 255)

    return np.array(train_x), np.array(train_y)


def image_generator(dataset, batch_size, ftc_preprocess):
    random.shuffle(dataset)
    current_index = 0
    while True:
        batch_content = dataset[current_index:current_index + batch_size]
        batch_content = ftc_preprocess(batch_content)
        yield batch_content
        current_index += batch_size
        if current_index > len(dataset):
            current_index = 0
            random.shuffle(dataset)


def main(args):
    global hr_input_dims

    epoch = args.epoch
    step_per_epoch = args.step
    batch_size = args.batch_size

    print("Getting dataset...")
    dataset = import_dataset(args.data_path, args.extension_file)

    dataset_iter = image_generator(dataset, batch_size, partial(preprocessing, lr_factor=lr_factor))

    discriminator_model = Discriminator(hr_input_dims,
                                        batch_size=batch_size, save_path=args.ckpnt_discr)
    if args.weights_discr_path:
        discriminator_model.load_weights(args.weights_discr_path)

    # Puisqu"il n'y a pas de Fully-Connected, le input shape peut varier.
    generator_model = Generator((None, None, 3), hr_input_dims,
                                batch_size=batch_size, nb_filter_conv1=16, save_path=args.ckpnt_gen)
    if args.weights_gen_path:
        generator_model.load_weights(args.weights_gen_path)

    print("Training is starting...")
    for e in range(epoch):
        for step in range(step_per_epoch):

            print(f"epoch {e}, step {step}...")

            train_X, train_Y = next(dataset_iter)
            img_outputs = generator_model.forward(train_X)

            disc_X = np.concatenate([train_Y, img_outputs])

            disc_Y = np.zeros(2 * batch_size)
            disc_Y[:batch_size] = 0.9
            disc_loss = discriminator_model.train(disc_X, disc_Y)

            generator_model.update_disc_loss(disc_loss[0])
            gen_loss = generator_model.train(train_X, train_Y)

            if step % 10 == 0:
                print(f"epoch: {e}, step: {step}")
                print(f"loss generator: {gen_loss}, loss dirscriminator: {disc_loss}")

        if args.ckpnt:
            print("Saving checkpoint...")
            discriminator_model.save()
            generator_model.save()


if __name__ == "__main__":
    args = parse_args()

    config.set(args)

    if config.location == "local":
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gcloud.json"

    main(args)
