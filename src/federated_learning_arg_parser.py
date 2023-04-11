from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class FederatedLearningArgParser(ArgumentParser):
    def __init__(self):
        super().__init__(
            prog="federated_learning.sh",
            description="Start federated learning",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--experiment",
            type=str,
            default="FedAvg",
            help="Additional code to help engineer distinguish multiple Federated Learning jobs from one another",
        )
        self.add_argument(
            "--name",
            type=str,
            default="sn",
            help="Name of the engineer who kicks of the federated learning job.",
        )
        self.add_argument(
            "--server_strategy",
            type=str,
            default="FedAvg",
            help="The weight averaging strategy used by the federated learning server.",
        )