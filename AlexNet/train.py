import torch
import torch.nn as nn
from model import Model
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrainModel(nn.Module):
    def __init__(self, momentum=0.9, batch_size=128, weight_decay=0.0005, learning_rate=0.01, epoch=90):
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss = nn.MSELoss()
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_loss, self.val_loss = [], []
        self.train_accuracy, self.val_accuracy = [], []

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
        
    def to_device(self, batch):
        images ,labels = batch
        return images.to(self.device), labels.to(self.device)
    
    