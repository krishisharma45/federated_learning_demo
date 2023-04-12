from typing import List, OrderedDict

import flwr as fl
import numpy as np
import torch


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, model):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model

    def get_parameters(self, config):
        return self.get_parameters(self.net)

    def fit(self, parameters, config):
        self.set_parameters(self.net, parameters)
        self.model.train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)
        loss, accuracy = self.model.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def get_parameters(self, net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict[{k: torch.Tensor(v) for k, v in params_dict}]
        net.load_state_dict(state_dict, strict=True)