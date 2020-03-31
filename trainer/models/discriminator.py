import keras.applications
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

from trainer.models.base_model import BaseModel


class Discriminator(BaseModel):

    def __init__(self, input_dims, save_path="save/discriminator", optimizer=Adam(learning_rate=0.0005)):
        super().__init__(save_path, optimizer)
        self.input_dims = input_dims
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        base_model = keras.applications.VGG16(input_shape=self.input_dims, include_top=False, pooling=None)
        classifier_part = Flatten()(base_model.layers[-1].output)
        classifier_part = Dense(512, activation="relu")(classifier_part)
        classifier_part = Dense(1, activation="sigmoid")(classifier_part)
        model = keras.Model(inputs=[base_model.input], outputs=[classifier_part])
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
