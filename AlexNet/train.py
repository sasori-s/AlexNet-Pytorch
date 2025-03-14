import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model
from preprocessing import LoadDataset
from tqdm import tqdm
from colorama import Fore, Style, init
import copy
from training_utils import TrainingUtils

init(autoreset=True)

class TrainModel(nn.Module):
    def __init__(self, data_path, momentum=0.9, batch_size=64, weight_decay=0.0005, learning_rate=0.1, epoch=90, num_classes=90, save_models=True, testing_mode=False):
        super(TrainModel, self).__init__()
        self.save_models = save_models
        self.testing_mode = testing_mode
        self.model_name = 'AlexNet'
        self.data_path = data_path
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.num_classes = num_classes
        self.utils = TrainingUtils()

        self.load_train_test_data()
        self.model = Model(num_classes=90)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.cuda_profile = next(self.model.parameters()).is_cuda
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss = nn.CrossEntropyLoss().cuda()
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_loss, self.val_loss = [], []
        self.train_accuracy, self.val_accuracy = [], []
        self.not_improved = 0
        self.patience = 10

    
    def load_train_test_data(self):
        data = LoadDataset(self.data_path, self.data_path, batch_size=self.batch_size, testing_mode=self.testing_mode, num_classes=self.num_classes)
        train_loader, test_loader = data.augment_dataset()
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.classes = test_loader.dataset.classes
        # print(f"{Fore.LIGHTMAGENTA_EX}  {train_loader.dataset.classes}")
        # print(f"{Fore.LIGHTMAGENTA_EX}  {test_loader.dataset.classes}")


        
    def to_device(self, batch):
        images ,labels = batch
        return images.to(self.device), labels.to(self.device)


    def train_epoch(self, epoch):
        self.model.train()
        current_accuracy, current_loss = 0.0, 0.0

        # initial_state = self.utils.save_current_parameters(self)

        for idx, batch in tqdm(enumerate(self.train_loader), desc='Training'):

            images, labels = self.to_device(batch)
            pred = self.model(images)
            loss = self.loss(pred, labels)

            # print(f"{Fore.YELLOW} The predicitons are {pred.shape} \n {pred.argmax(1)} {Fore.CYAN} \n The true labeles are {labels.unsqueeze(1).shape}\n {labels}")
            # print("\033[92m [LOSS INFO {}th iteration] The training loss is {:.3f} \033[0m".format(idx, loss.item()))

            
            # self.utils.call_backward_hook(self)
            loss.backward(retain_graph=True)
            # print("-------------------ENd of the hooks-------------------")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()
            self.optimizer.zero_grad()
            current_loss += loss.item()
            current_accuracy += (pred.argmax(1) == labels).float().mean().item()
            # print("\033[101m [ACCURACY INFO {}th iteration] The training accuracy is {} \033[0m".format(idx, (pred.argmax(1) == labels).float().mean().item()))
        
        # self.utils.plot_grad_flow(self, self.model.named_parameters(), epoch,  idx)

        self.model.activations['conv1'].register_hook(lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of relu1 is {grad.mean()}"))
        self.model.activations['conv5'].register_hook(lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of relu_fc1 is {grad.mean()}"))

        current_loss /= len(self.train_loader)
        current_accuracy /= len(self.train_loader)

        print("\033[101m [TRAINING INFO ==> {} epoch] \033[0m".format(epoch))
        print(f"{Fore.LIGHTGREEN_EX} The current training loss is {current_loss:.3f}")
        print(f"{Fore.LIGHTBLUE_EX} The current trianing accuracy is {current_accuracy:.3f}")

        self.train_accuracy.append(current_accuracy)
        self.train_loss.append(current_loss)
        
        # self.utils.compare_parameters(self, initial_state)

        return current_loss, current_accuracy
        #Do not forget to include the scheduler.step() in the run funciton. 
            
    
    def validate_epoch(self, epoch):
        self.model.eval()
        current_loss = 0.0
        current_accuracy = 0.0

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_loader), desc='Validation'):
                images, labels = self.to_device(batch)
                pred = self.model(images)
                loss = self.loss(pred, labels)
                # print(Fore.LIGHTRED_EX + "[LOSS INFO {}th iteration] The validation loss is {:.3f}".format(idx, loss.item()) + Style.RESET_ALL)
                # print(Fore.BLUE + "The predicitons are {}".format(pred.argmax(1)) + Fore.GREEN + "\n The true labels are {}".format(labels))
                current_loss += loss.item()
                current_accuracy += (pred.argmax(1) == labels).float().mean().item()
                # print(Fore.LIGHTRED_EX + "[ACCURACY INFO {}th iteration] The validation accuracy is {:.3f}".format(idx, (pred.argmax(1) == labels).float().mean().item()) + Style.RESET_ALL)

            current_loss /= len(self.test_loader)
            current_accuracy /= len(self.test_loader)

            print("\033[101m [VALIDATION INFO ==> {} epoch] \033[0m".format(epoch))
            print(f"{Fore.LIGHTMAGENTA_EX} The current validation loss is ==> {current_loss:.3f}")
            print(f"{Fore.LIGHTYELLOW_EX} The current validation accuracy is ==> {current_accuracy:.3f}")
            
            self.val_accuracy.append(current_accuracy)
            self.val_loss.append(current_loss)

            return current_loss, current_accuracy
        

    
    def run(self):
        print("\033[92m [INFO] Starting the tranining process \033[0m ")
        for epoch in range(1, self.epoch + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            self.scheduler.step(val_loss)
            if self.save_models:
                self.utils.save_best_model(self, val_loss, val_acc)

            self.utils.verbose(self, epoch, train_loss, train_acc, val_loss, val_acc)

        print("\033[92m [INFO] Training has been completed \033[0m")
        self.utils.save_final_model(self)
        return self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy
    

    def __call__(self):
        return self.run()


if __name__ == '__main__':
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
    train_model = TrainModel(data_path, save_models=True, num_classes=10, testing_mode=True)
    train_model()
