import tensorflow as tf
from keras import Model
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, BatchNormalization, Input, Add
from keras.optimizers import Adam

from trainer.models.base_model import BaseModel


def penalize_loss(disc_loss):
    def mse_loss(predict_y, target_y):
        square_error = tf.math.reduce_mean(tf.square(predict_y - target_y))
        # square_error = square_error / tf.cast(tf.shape(predict_y)[0], dtype=tf.float32)
        return square_error + disc_loss
    return mse_loss


class Generator(BaseModel):

    def __init__(self, save_path="save/generator", optimizer=Adam(learning_rate=0.0005)):
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
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = BatchNormalization()(x)

        # Reconstruire les features dans le SR
        y = Conv2D(64, kernel_size=5, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        y = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(y)

        shortcut = Conv2D(64, 1, strides=1, activation="relu")(x)
        y = Add()([y, shortcut])
        y = BatchNormalization()(y)

        y = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(y)

        # output_img va etre 1 fm si gray scale et 3 fm si couleur.
        output_img = Conv2D(3, kernel_size=5, strides=1, activation="relu", padding="same",
                            kernel_initializer=VarianceScaling(scale=2))(y)

        # fm = MaxPool2D(3, strides=1)(output_img)

        model = Model(inputs=[inputs], outputs=[output_img])  # outputs=[output_img, fm]

        return model

    def compile(self):
        # Todo: changer pour loss sur fm
        self.model.compile(loss=penalize_loss(self.discr_loss), optimizer=self.optimizer, metrics=["mse"])

    def update_disc_loss(self, loss):
        """
        On prend le log parce que si le discriminateur distingue le genere du non, on veut amplifier cette erreur et
        s'il n'arrive pas a distinguer on veut minimiser la correction a apporter.
        """
        loss = tf.cast(-tf.math.log(loss*0.1), dtype=tf.float32)
        self.discr_loss.assign(loss)

    @classmethod
    def load(cls, load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        gen = Generator(save_path=save_path)
        gen.model.load_weights(load_path)
        gen.compile()
        return gen
