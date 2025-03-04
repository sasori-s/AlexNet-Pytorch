from local_response_normalization import LocalResponseNormalization
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from torch.nn import LocalResponseNorm
from colorama import Fore, Style, init

init(autoreset=True)

class Model(nn.Module):
    def __init__(self, num_classes=90):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.norm1 = LocalResponseNormalization(5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.norm2 = LocalResponseNormalization(5)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=False)
        self.activations = {}


    def forward_(self, x):
        x = self.conv1(x)
        print(f"{Fore.LIGHTMAGENTA_EX} Requires_grad --> {x.requires_grad} The gradient of first reLU activation is : {x.grad}")
        x = self.relu(x)
        self.activations['relu0'] = x.detach()
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        self.activations['relu1'] = x.detach()
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        self.activations['relu2'] = x.detach()

        x = self.conv4(x)
        x = self.relu(x)
        self.activations['relu3'] = x.detach()

        x = self.conv5(x)
        x = self.relu(x)
        self.activations['relu4'] = x.detach()
        x = self.pool3(x)
        # print(f"{Fore.LIGHTMAGENTA_EX} The shape of x before fc layer is :  {x.shape}")
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        self.activations['relu5'] = x.detach()
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        self.activations['relu6'] = x.detach()
        x = self.dropout(x)

        x = self.fc3(x)
        return x

    def forward(self, input):
        conv1 = self.conv1(input)
        relu1 = self.relu(conv1)
        self.relu1_grad = relu1 
        norm1 = self.norm1(relu1)
        pool1 = self.pool1(norm1)

        conv2 = self.conv2(pool1)
        relu2 = self.relu(conv2)
        norm2 = self.norm2(relu2)
        pool2 = self.pool2(norm2)

        conv3 = self.conv3(pool2)
        relu3 = self.relu(conv3)

        conv4 = self.conv4(relu3)
        relu4 = self.relu(conv4)

        conv5 = self.conv5(relu4)
        relu5 = self.relu(conv5)
        pool5 = self.pool3(relu5)

        flatten = pool5.view(pool5.size(0), -1)

        fc1 = self.fc1(flatten)
        relu_fc1 = self.relu(fc1)
        dropout_fc1 = self.dropout(relu_fc1)

        fc2 = self.fc2(dropout_fc1)
        relu_fc2 = self.relu(fc2)
        dropout_fc2 = self.dropout(relu_fc2)

        fc3 = self.fc3(dropout_fc2)
        return fc3



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("\tTHe device is {}".format(device))
    model = Model()
    model.to(device)
    input = torch.randn(32, 3, 224, 224).to(device)
    oouput = model(input)
    print(summary(model, (3, 224, 224)))
    print(f"{Fore.LIGHTGREEN_EX} The gradient is : {oouput.grad}")
    
    for key, value in model.activations.items():
        print(f"{Fore.LIGHTCYAN_EX} The shape of {key} is : {value.shape}")
        print(f"{Fore.LIGHTCYAN_EX} The gradient of {key} is : {value.grad}")