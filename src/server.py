import flwr as fl

from src.server_config import ServerConfig

if __name__ == "__main__":
    server_config = ServerConfig()
    fl.server.start_server(server_address="0.0.0.0:8080",
                           config={**server_config.__dict__},
                           strategy=fl.server.strategy.FedAvg())
