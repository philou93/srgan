import tensorflow as tf
from keras import Model
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, BatchNormalization, Input
from keras.optimizers import Adam
from tensorflow import Variable
import keras.backend as K

from trainer.models.base_model import BaseModel


def penalize_loss(disc_loss):
    def mse_loss(predict_y, target_y):
        square_error = K.mean(tf.square(predict_y - target_y), axis=-1)
        return square_error + disc_loss
    return mse_loss


class Generator(BaseModel):

    def __init__(self, save_path="save/generator", optimizer=Adam(learning_rate=0.0001)):
        super().__init__(save_path, optimizer)
        self.discr_loss = tf.Variable(0.0)
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        inputs = Input(shape=(None, None, 3))
        # Couche pour extraire les feature de l'image LR
        x = Conv2D(128, kernel_size=9, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(inputs)
        x = Conv2D(128, kernel_size=9, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(inputs)
        x = BatchNormalization()(x)

        # Reconstruire les features dans le SR
        y = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(y)
        y = BatchNormalization()(y)

        # output_img va etre 1 fm si gray scale et 3 fm si couleur.
        output_img = Conv2D(3, kernel_size=5, strides=1, activation="relu", padding="same",
                            kernel_initializer=VarianceScaling(scale=2))(y)

        # fm = MaxPool2D(3, strides=1)(output_img)

        model = Model(inputs=[inputs], outputs=[output_img])  # outputs=[output_img, fm]
        # model.add_loss(self.disc_loss)  # TODO: est-ce que ca marche vraiment???

        return model

    def compile(self):
        # Todo: changer pour loss sur fm
        self.model.compile(loss=penalize_loss(self.discr_loss), optimizer=self.optimizer, metrics=["mse"])

    def update_disc_loss(self, loss):
        loss = abs(loss * 0.001)
        self.discr_loss.assign(loss)
        tf.print(self.discr_loss)

    @classmethod
    def load(cls, nb_filter_conv1=32, load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        gen = Generator(save_path=save_path)
        gen.model.load_weights(load_path)
        gen.compile()
        return gen
