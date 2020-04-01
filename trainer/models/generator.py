import tensorflow as tf
from keras import Model
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, BatchNormalization, Input, Add, ReLU
from keras.optimizers import Adam
from tensorflow import Variable

from trainer.models.base_model import BaseModel


class Generator(BaseModel):

    def __init__(self, nb_filter_conv1=32, save_path="save/generator", optimizer=Adam(learning_rate=0.0001)):
        super().__init__(save_path, optimizer)
        self.nb_filer_conv1 = nb_filter_conv1
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        self.disc_loss = Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

        inputs = Input(shape=(None, None, 3))
        x = Conv2D(self.nb_filer_conv1, kernel_size=9, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(inputs)
        x = Conv2D(self.nb_filer_conv1, kernel_size=9, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = Conv2D(self.nb_filer_conv1, kernel_size=9, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        y = Conv2D(self.nb_filer_conv1 * 2, kernel_size=7, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(self.nb_filer_conv1 * 2, kernel_size=7, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        shortcut = Conv2D(self.nb_filer_conv1 * 2, kernel_size=1, strides=1, padding="same",
                          kernel_initializer=VarianceScaling(scale=2))(x)
        x = Add()([y, shortcut])

        y = Conv2D(self.nb_filer_conv1 * 4, kernel_size=5, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(self.nb_filer_conv1 * 4, kernel_size=5, strides=1, padding="same",
                   kernel_initializer=VarianceScaling(scale=2))(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        shortcut = Conv2D(self.nb_filer_conv1 * 4, kernel_size=1, strides=1, padding="same",
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
        output_img = Conv2D(3, kernel_size=1, strides=1,
                            kernel_initializer=VarianceScaling(scale=2), activation="softmax")(y)

        # fm = MaxPool2D(3, strides=1)(output_img)

        model = Model(inputs=[inputs], outputs=[output_img])  # outputs=[output_img, fm]
        # model.add_loss(self.disc_loss)  # TODO: est-ce que ca marche vraiment???

        return model

    def compile(self):
        # Todo: changer pour loss sur fm
        self.model.compile(loss="mae", optimizer=self.optimizer, metrics=["mae"])

    def update_disc_loss(self, loss):
        loss = abs(loss * 0.001)
        self.disc_loss.assign(loss)

    @classmethod
    def load(cls, nb_filter_conv1=32, load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        gen = Generator(nb_filter_conv1=nb_filter_conv1, save_path=save_path)
        gen.model.load_weights(load_path)
        gen.compile()
        return gen
