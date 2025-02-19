import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model
from preprocessing import LoadDataset
from tqdm import tqdm

class TrainModel(nn.Module):
    def __init__(self, data_path, momentum=0.9, batch_size=128, weight_decay=0.0005, learning_rate=0.01, epoch=90, num_classes=90):
        super(TrainModel, self).__init__()
        self.data_path = data_path
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.num_classes = num_classes

        self.load_train_test_data()
        self.model = Model()
        self.initialize_weight_and_bias()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss = nn.MSELoss()
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_loss, self.val_loss = [], []
        self.train_accuracy, self.val_accuracy = [], []

    
    def load_train_test_data(self):
        data = LoadDataset(self.data_path, self.data_path)
        train_loader, test_loader = data.augment_dataset()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = train_loader.dataset.classes

    
    def initialize_weight_and_bias(self):
        bias_to_zero = ['conv1', 'conv3']
        bias_to_one = ['conv2', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.01)

            if 'bias' in name:
                if name in bias_to_zero:
                    param.data.fill_(0)
                elif name in bias_to_one:
                    param.data.fill_(1)
                else:
                    param.data.fill_(0)

        
    def to_device(self, batch):
        images ,labels = batch
        return images.to(self.device), labels.to(self.device)
    

    def train_epoch(self):
        self.model.train()
        current_accuracy, current_loss = 0.0, 0.0

        for idx, batch in tqdm(enumerate(self.train_loader), desc='Training'):
            images, labels = self.to_device(batch)
            pred = self.model(images)
            loss = self.loss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            current_loss += loss.item()
            current_accuracy += (pred.argmax(1) == labels).float().mean().item() 
        
        current_loss /= len(self.train_loader)
        current_accuracy /= len(self.train_loader)
        return current_loss, current_accuracy
        #Do not forget to include the scheduler.step() in the run funciton. 
            
    
    def validate_epoch(self):
        pass



if __name__ == '__main__':
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
    train_model = TrainModel(data_path)
    train_model.train_epoch()
