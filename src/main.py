from typing import List, OrderedDict
import numpy as np
import flwr as fl
import torch

from src.client import FlowerClient
from src.federated_learning_arg_parser import FederatedLearningArgParser
from src.training.dataloader import DEVICE, trainloaders, valloaders
from src.training.model import Net


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, model)

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)


if __name__ == "__main__":
    args = FederatedLearningArgParser().parse_args()

    if args.server_strategy in ["FaultTolerantFedAvg", "FedAdagrad", "FedAdam", "FedYogi", "FedAvgM"]:
        model_file = "./data/convnext.h5"
        model = load_model(model_file, custom_objects={"MulticlassAUC": MulticlassAUC,
                                                       "ConvNeXtModel": ConvNeXtModel,
                                                       "DenseMonotone": DenseMonotone,
                                                       })

    fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)
