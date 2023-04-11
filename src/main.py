import flwr as fl

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow.keras.models import load_model


class FlowerServerArgParser(ArgumentParser):
    def __init__(self):
        super().__init__(
            prog="flower.sh",
            description="Train with federated learning",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--server_strategy",
            type=str,
            default="FedAvg",
            help="The weight averaging strategy used by the federated learning server"
        )


if __name__ == "__main__":

    args = FlowerServerArgParser().parse_args()

    if args.server_strategy in ["FaultTolerantFedAvg", "FedAdagrad", "FedAdam", "FedYogi", "FedAvgM"]:
        model_file = "./data/convnext.h5"
        model = load_model(model_file, custom_objects={"MulticlassAUC": MulticlassAUC,
                                                       "ConvNeXtModel": ConvNeXtModel,
                                                       "DenseMonotone": DenseMonotone,
                                                       })

    if args.server_strategy == "FedAvg":
        strategy = fl.server.strategy.FedAvg()
    elif args.server_strategy == "FastAndSlow":
        strategy = fl.server.strategy.FastAndSlow(min_fit_clients=2, min_eval_clients=2, min_available_clients=2, importance_sampling=False)
    elif args.server_strategy == "FedAvgM":
        strategy = fl.server.strategy.FedAvgM(initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
    elif args.server_strategy == "FaultTolerantFedAvg":
        strategy = fl.server.strategy.FaultTolerantFedAvg(
            initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
    elif args.server_strategy == "FedAdam":
        strategy = fl.server.strategy.FedAdam(initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
    elif args.server_strategy == "FedAdagrad":
        strategy = fl.server.strategy.FedAdagrad(
            initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
    elif args.server_strategy == "FedYogi":
        strategy = fl.server.strategy.FedYogi(initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
    else:
        raise KeyError(f"Strategy {args.server_strategy} not recognized.")

    fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)
