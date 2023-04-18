import flwr as fl
import numpy as np

# import tensorflow as tf

from tensorflow.keras.datasets import mnist

from src.model import CreateMnistModel
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

model = CreateMnistModel().run()


class MnistClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        print("Epoch fitting complete.")
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


fl.client.start_numpy_client(server_address="flower_federated_learning_demo:8080", client=MnistClient())
