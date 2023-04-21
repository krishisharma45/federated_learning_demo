import flwr as fl

from src.config.server_config import ServerConfig, Config

if __name__ == "__main__":
    server_config = ServerConfig()
    fl.server.start_server(server_address=f"{server_config.server_address}:8080",
                           config={**Config().__dict__},
                           strategy=fl.server.strategy.FedAvg(min_available_clients=server_config.min_available_clients))