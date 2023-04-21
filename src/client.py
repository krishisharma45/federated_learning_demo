import sys
import flwr as fl

from src.config.client_config import ClientConfig
from src.training.client import MnistClient


client_config = ClientConfig(device_id=sys.argv[1])
client = MnistClient(client_config=client_config)
fl.client.start_numpy_client(server_address=f"{client_config.server_address}:8080", client=client)