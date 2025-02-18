import torch
import torch.nn as nn
from model import Model

class TrainModel(nn.Module):
    def __init__(self, momentum=0.9, batch_size=128, weight_decay=0.0005, learning_rate=0.01):
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss = nn.MSELoss()
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.best_loss = float('inf')
        self.best_accuracy = 0.0

    def initialize_weight_and_bias(self):
        for m in self.model.modules():
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)

        self.model.conv1.bias.data.fill(0)
        self.model.conv2.bias.data.fill(1)
        self.model.conv3.bias.data.fill(0)
        self.model.conv4.bias.data.fill(1)
        self.model.conv5.bias.data.fill(1)
        self.model.fc1.bias.data.fill(1)
        self.model.fc2.bias.data.fill(1)
        self.model.fc3.bias.data.fill(1)
        