import sys
import flwr as fl

from src.client_config import ClientConfig
from src.dataset import Dataset
from src.model import CreateMnistModel


class MnistClient(fl.client.NumPyClient):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.model = CreateMnistModel().run()
        self.x_train, self.x_test, self.y_train, self.y_test = Dataset(device_id=device_id).get()

    def get_parameters(self):
        print("---GET PARAMETERS---")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("---FIT---")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        print("Epoch fitting complete.")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("---EVALUATE---")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


if __name__ == "__main__":
    client_config = ClientConfig(device_id=sys.argv[1])
    client = MnistClient(device_id=client_config.device_id)
    print("---CLIENT---")
    print(client)
    fl.client.start_numpy_client(server_address=f"{client_config.server_address}:8080",
                                 client=client)
