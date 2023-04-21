from typing import List, Tuple, Any

import flwr as fl

from src.config.server_config import ServerConfig, Config


class FlowerServer:
    def __init__(self, server_config: ServerConfig):
        self.server_config = server_config

    def run(self):
        fl.server.start_server(server_address=f"{self.server_config.server_address}:8080",
                               config={**Config().__dict__},
                               strategy=self._strategy())

    def _strategy(self):
        return fl.server.strategy.FedAvg(
            min_available_clients=self.server_config.min_available_clients,
            min_fit_clients=self.server_config.min_fit_clients,
            min_eval_clients=self.server_config.min_eval_clients,
            evaluate_metrics_aggregation_fn=self.weighted_average
        )

    @staticmethod
    def weighted_average(metrics: List[Tuple[int, Any]]) -> Any:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    server_config = ServerConfig()
    flower_server = FlowerServer(server_config=server_config)
    flower_server.run()
