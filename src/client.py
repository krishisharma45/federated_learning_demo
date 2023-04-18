import flwr as fl
import numpy as np
from tensorflow.keras.datasets import mnist
from src.model import CreateMnistModel


class MnistClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CreateMnistModel().run()
        self.x_train, self.x_test, self.y_train, self.y_test = self._normalize()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        print("Epoch fitting complete.")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def _normalize(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        return x_train, x_test, y_train, y_test


fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
