import keras.applications
from keras.layers import Flatten, Dense, Input, Conv2D, BatchNormalization, MaxPool2D
from keras.initializers import VarianceScaling
from keras.optimizers import Adam

from trainer.models.base_model import BaseModel


class Discriminator(BaseModel):

    def __init__(self, input_dims, save_path="save/discriminator", optimizer=Adam(learning_rate=0.0005)):
        super().__init__(save_path, optimizer)
        self.input_dims = input_dims
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        # base_model = keras.applications.VGG16(input_shape=self.input_dims, include_top=False, pooling=None)

        # Petit VGG
        inputs = Input(shape=self.input_dims)
        x = Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(inputs)
        x = Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D()(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu",
                   kernel_initializer=VarianceScaling(scale=2))(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        classifier_part = Flatten()(x)
        classifier_part = Dense(1024, activation="relu")(classifier_part)
        classifier_part = Dense(512, activation="relu")(classifier_part)
        classifier_part = Dense(1, activation="softmax")(classifier_part)
        model = keras.Model(inputs=[inputs], outputs=[classifier_part])
        return model

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["acc"])

    @classmethod
    def load(cls, input_dims, optimizer=Adam(learning_rate=0.0005),
             load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        discr = Discriminator(input_dims, save_path=save_path, optimizer=optimizer)
        discr = discr.model.load_weights(load_path)
        discr.compile()
        return discr
