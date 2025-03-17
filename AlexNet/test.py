import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        pass

    def load_model(self, model_path):
        pass
        
