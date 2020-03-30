import tensorflow as tf
from keras import Model
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, BatchNormalization, Input, Add, ReLU
from keras.optimizers import Adam
from keras.backend import log as k_log
from tensorflow import Variable
from tensorflow.python.lib.io import file_io

from trainer.utils.download import split_bucket, download_weight


class Generator:

    def __init__(self, input_dims, output_dims, batch_size=32,
                 nb_filter_conv1=32, save_path="save/generator"):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.nb_filer_conv1 = nb_filter_conv1
        self.batch_size = batch_size
        self.save_path = save_path
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        self.disc_loss = Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

        inputs = Input(shape=self.input_dims)
        x = Conv2D(self.nb_filer_conv1, kernel_size=9, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(inputs)
        x = Conv2D(self.nb_filer_conv1, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = Conv2D(self.nb_filer_conv1, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        y = Conv2D(self.nb_filer_conv1 * 2, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(self.nb_filer_conv1 * 2, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        shortcut = Conv2D(self.nb_filer_conv1 * 2, kernel_size=1, strides=1, padding="same",
                          kernel_initializer=VarianceScaling(scale=2))(x)
        x = Add()([y, shortcut])

        y = Conv2D(self.nb_filer_conv1 * 4, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(self.nb_filer_conv1 * 4, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        shortcut = Conv2D(self.nb_filer_conv1 * 4, kernel_size=1, strides=1, padding="same",
                          kernel_initializer=VarianceScaling(scale=2))(x)
        y = Add()([y, shortcut])

        # output_img va etre 1 fm si gray scale et 3 fm si couleur.
        output_img = Conv2D(self.output_dims[-1], kernel_size=1, strides=1,
                            kernel_initializer=VarianceScaling(scale=2))(y)

        # fm = MaxPool2D(3, strides=1)(output_img)

        model = Model(inputs=[inputs], outputs=[output_img])  # outputs=[output_img, fm]
        model.add_loss(self.disc_loss)  # TODO: est-ce que ca marche vraiment???

        return model

    def compile(self):
        # Todo: changer pour loss sur fm
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])

    def update_disc_loss(self, loss):
        loss = loss * 0.001
        self.disc_loss.assign(loss)

    def forward(self, inputs):
        return self.model.predict(inputs)

    def train(self, X, Y):
        loss = self.model.train_on_batch(X, Y)
        return loss

    def save(self):
        """
        Fonction pour google cloud storage.
        """
        if self.save_path.startswith("gs://"):  # on l'enregistre sur google cloud storage
            self.model.save('model.h5')  # on l'enregistre temporairement
            with file_io.FileIO('model.h5', mode='rb') as input_f:
                with file_io.FileIO(self.save_path, mode='wb+') as output_f:
                    output_f.write(input_f.read())
        else:
            self.model.save(self.save_path, "model.h5")

    def load_weights(self, path):
        if path.startswith("gs://"):
            from google.cloud import storage
            bucket_name, sub_folder = split_bucket(path)
            storage_client = storage.Client()
            tmp_file = download_weight(bucket_name, sub_folder, storage_client)
            self.model.load_weights(tmp_file)  # load_weights veut un f***** path
            file_io.delete_file(tmp_file)
        else:
            self.model.load_weights(path)
        self.compile()

    @classmethod
    def load(cls, input_dims, output_dims, batch_size=32, nb_filter_conv1=32,
             load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        gen = Generator(input_dims, output_dims, batch_size=batch_size,
                        nb_filter_conv1=nb_filter_conv1, save_path=save_path)
        gen.model.load_weights(load_path)
        gen.compile()
        return gen
