import os
import random
from functools import partial

import numpy as np
import tensorflow as tf

import trainer.config as config
from trainer.argsparser import parse_args
from trainer.models.discriminator import Discriminator
from trainer.models.generator import Generator
from trainer.utils.calcul_util import preprocess_imgs
from trainer.utils.download import split_bucket, download_images_async
from trainer.utils.gpus import setup_device_use, set_to_memory_growth
from trainer.utils.loss_history import add_loss_to_history, flush_loss_history

hr_input_dims = [400, 400, 3]  # Hardcoder


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
    """
    On garde en memoire seulement les images quon utilisepour l'époque courrante.

    :param blobs:
    :param lr_factor:
    :param extension:
    :return:
    """
    train_x = []
    train_y = []
    images = download_images_async(blobs, extension_file=extension)
    for image in images:
        # C'est important pour le training que toutes les images ait le même shape à l'entré.
        image_input, image_output = preprocess_imgs(image, hr_input_dims, lr_factor, pad=True)
        train_x.append(image_input)
        train_y.append(image_output)

    return np.array(train_x), np.array(train_y)


def image_generator(dataset, batch_size, ftc_preprocess):
    random.shuffle(dataset)
    current_index = 0
    while True:
        batch_content = dataset[current_index:current_index + batch_size]
        batch_content = ftc_preprocess(batch_content)
        if len(batch_content[0].shape) != 4 or len(batch_content[1].shape) != 4:
            print(f"The shape of the dataset is not valid. Got datasets shape of {len(batch_content[0])} and "
                  f"{len(batch_content[1])} elements, but 4 are needed.")
        else:
            yield batch_content
        current_index += batch_size
        if current_index > len(dataset) - batch_size:
            current_index = 0
            random.shuffle(dataset)


def main(args):
    global hr_input_dims

    # TODO: il faudrait enregistrer les parametres pour reconstruire le model

    epoch = args.epoch
    step_per_epoch = args.step
    batch_size = args.batch_size

    print("Getting dataset...")
    dataset = import_dataset(args.data_path, args.extension_file)

    dataset_iter = image_generator(dataset, batch_size, partial(preprocessing, lr_factor=args.lr_factor))

    discriminator_model = Discriminator(hr_input_dims, save_path=args.ckpnt_discr)
    if args.weights_discr_path:
        discriminator_model.load_weights(args.weights_discr_path)

    generator_model = Generator(save_path=args.ckpnt_gen)
    if args.weights_gen_path:
        generator_model.load_weights(args.weights_gen_path)

    print("Training is starting...")
    for e in range(epoch):

        generator_model.reset_optimizer()
        discriminator_model.reset_optimizer()

        for step in range(step_per_epoch):
            train_X, train_Y = next(dataset_iter)

            with tf.device(args.gpus_mapper["generator"]):
                img_outputs = generator_model.forward(train_X)

            disc_X = np.concatenate([train_Y, img_outputs])

            disc_Y = np.zeros(2 * batch_size)
            """
            On ne met pas 1 pour les images générées parce que même si le modèle comprend que cest pas une vrai 
            image, on veut lui donner un peu de difficulté (il va toujours avoir un doute si cest faut).
            """
            disc_Y[:batch_size] = 0.95

            with tf.device(args.gpus_mapper["descriminator"]):
                disc_loss = discriminator_model.train(disc_X, disc_Y)

            generator_model.update_disc_loss(disc_loss)
            with tf.device(args.gpus_mapper["generator"]):
                gen_loss = generator_model.train(train_X, train_Y)

            add_loss_to_history(gen_loss, disc_loss)

            if step % int(step_per_epoch / 2) == 0:
                print(f"epoch: {e}, step: {step}")
                print(f"Generator loss: {gen_loss}  --  Discriminator loss: {disc_loss}")

        if args.ckpnt:
            print("Saving checkpoint...")
            discriminator_model.save()
            generator_model.save()
            flush_loss_history(args.history_path)


if __name__ == "__main__":
    args = parse_args()

    config.set(args)

    # Sur gcloud la variable existe déjà
    if config.location == "local":
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gcloud.json"

    gpus_mapper = setup_device_use()
    set_to_memory_growth()
    setattr(args, "gpus_mapper", gpus_mapper)

    main(args)
