import torch
import torch.nn as nn
from model import Model

class TrainModel(nn.Module):
    def __init__(self):
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss = nn.MSELoss()