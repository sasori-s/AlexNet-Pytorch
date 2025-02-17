from local_response_normalization import LocalResponseNormalization
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

class Model(nn.Module):
    def __init__(self, num_classes=90):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.norm1 = LocalResponseNormalization()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.norm2 = LocalResponseNormalization()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 9 * 9, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = Model()
    model.to(device)
    input = torch.randn(1, 3, 227, 227).to(device)
    model(input)
    print(summary(model, (3, 227, 227), device='cpu'))