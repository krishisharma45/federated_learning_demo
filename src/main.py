import flwr as fl
from tensorflow.keras.models import load_model
from src.federated_learning_arg_parser import FederatedLearningArgParser


if __name__ == "__main__":
    args = FederatedLearningArgParser().parse_args()

    if args.server_strategy in ["FaultTolerantFedAvg", "FedAdagrad", "FedAdam", "FedYogi", "FedAvgM"]:
        model_file = "./data/convnext.h5"
        model = load_model(model_file, custom_objects={"MulticlassAUC": MulticlassAUC,
                                                       "ConvNeXtModel": ConvNeXtModel,
                                                       "DenseMonotone": DenseMonotone,
                                                       })

    fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)
