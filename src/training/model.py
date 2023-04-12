import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from flwr.common import Metrics
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.training.dataloader import DEVICE


class Net(nn.Module):
    def __init__(self, trainloaders, valloaders, testloader) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloader = testloader

    def run(self):
        trainloader = self.trainloaders[0]
        valloader = self.valloaders[0]
        net = Net().to(DEVICE)

        for epoch in range(5):
            self.train(net, trainloader, 1)
            loss, accuracy = self.test(net, valloader)
            print(f"Epoch {epoch + 1}: validation loss {loss}, accuracy {accuracy}")

        loss, accuracy = self.test(net, self.testloader)
        print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, net, trainloader, epochs: int, verbose=False):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self, net, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy