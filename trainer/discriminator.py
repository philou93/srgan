import keras.applications
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from tensorflow.python.lib.io import file_io

from trainer.utils.download import split_bucket, download_weight


class Discriminator:

    def __init__(self, input_dims, batch_size=32, nb_rb=4,
                 save_path="save/discriminator"):
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.save_path = save_path
        self.nb_rb = nb_rb
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        base_model = keras.applications.ResNet50V2(input_shape=self.input_dims, include_top=False, pooling=None)
        model = Sequential([base_model,
                            Flatten(),
                            Dense(512, activation="relu"),
                            Dense(1, activation="sigmoid")])
        return model

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["acc"])

    def forward(self, inputs):
        return self.model.predict(inputs)

    def train(self, X, Y):
        loss = self.model.train_on_batch(X, Y)
        return loss

    def save(self):
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
    def load(cls, input_dims, batch_size=32, nb_filter_conv1=16,
             load_path="save/discriminator/model.h5", save_path="save/discriminator"):
        discr = Discriminator(input_dims, batch_size=batch_size, save_path=save_path)
        discr = discr.model.load_weights(load_path)
        discr.compile()
        return discr
