import tensorflow as tf
from keras import Model
from keras.initializers import VarianceScaling
from keras.layers import Conv2D, BatchNormalization, Input, Add, ReLU
from keras.optimizers import Adam
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
        self.optimizer = None
        self.model = self.create_model()
        self.compile()

    def create_model(self):
        raise NotImplemented(f"'create_model' function must be implemented")

    def compile(self):
        raise NotImplemented(f"'compile' function must be implemented")

    def forward(self, inputs):
        return self.model.predict(inputs)

    def train(self, X, Y):
        loss = self.model.train_on_batch(X, Y)[0]
        return loss

    def reset_optimizer(self):
        self.optimizer.decay.assign(0.0)

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
    def load(*args, wkargs):
        raise NotImplemented("'load' function must be implemented.")
