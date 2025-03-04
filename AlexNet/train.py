import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model
from preprocessing import LoadDataset
from tqdm import tqdm
from colorama import Fore, Style, init
import copy

init(autoreset=True)

class TrainModel(nn.Module):
    def __init__(self, data_path, momentum=0.9, batch_size=64, weight_decay=0.0005, learning_rate=0.001, epoch=90, num_classes=90):
        super(TrainModel, self).__init__()
        self.model_name = 'AlexNet'
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
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_loss, self.val_loss = [], []
        self.train_accuracy, self.val_accuracy = [], []
        self.not_improved = 0
        self.patience = 10

    
    def load_train_test_data(self):
        data = LoadDataset(self.data_path, self.data_path, batch_size=self.batch_size, testing_mode=True)
        train_loader, test_loader = data.augment_dataset()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = test_loader.dataset.classes
        # print(f"{Fore.LIGHTMAGENTA_EX}  {train_loader.dataset.classes}")
        # print(f"{Fore.LIGHTMAGENTA_EX}  {test_loader.dataset.classes}")

    
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
    

    def see_grad(self):
        for name, reLU in self.model.activations.items():
            handle = self.model.activations[name].register_hook(
                lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of {name} is {grad.sum()}")
            )
            handle.remove()
    
    @staticmethod
    def backward_hook(model_layer, grad_input, grad_output):
        print(f"{Fore.LIGHTRED_EX} The layer is {model_layer} \n{Fore.BLUE} Input has {len(grad_input)} elements \n{Fore.CYAN} Output has {len(grad_output)} elements")

        for grad in grad_input:
            try:
                print(f" for input gradient {grad.mean()}")
            except AttributeError:
                print(f" None found for gradient")

        for grad in grad_output:
            try:
                print(f" for output gradient {grad.mean()}")
            except AttributeError:
                print(f" None found for gradient")
        


    def train_epoch(self):
        self.model.train()
        current_accuracy, current_loss = 0.0, 0.0

        initial_state = self.save_current_parameters()

        for idx, batch in tqdm(enumerate(self.train_loader), desc='Training'):
            self.optimizer.zero_grad()

            images, labels = self.to_device(batch)
            pred = self.model(images)
            loss = self.loss(pred, labels)

            print(f"{Fore.YELLOW} The prediciton shape is {pred.argmax(1)} {Fore.CYAN} \n The true label shape is {labels}")
            print("\033[92m [LOSS INFO {}th iteration] The training loss is {:.3f} \033[0m".format(idx, loss.item()))

            # self.see_grad()
            # self.model.activations['conv2'].register_hook(lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of relu1 is {grad.mean()}"))
            # self.model.activations['relu_fc1'].register_hook(lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of relu_fc1 is {grad.mean()}"))
            print("-------------------Calling the hooks-------------------")
            self.model.relu.register_full_backward_hook(self.backward_hook)

            loss.backward(retain_graph=True)
            print("-------------------ENd of the hooks-------------------")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()
            current_loss += loss.item()
            current_accuracy += (pred.argmax(1) == labels).float().mean().item()
            print("\033[101m [ACCURACY INFO {}th iteration] The training accuracy is {} \033[0m".format(idx, (pred.argmax(1) == labels).float().mean().item()))
        
        current_loss /= len(self.train_loader)
        current_accuracy /= len(self.train_loader)

        self.train_accuracy.append(current_accuracy)
        self.train_loss.append(current_loss)
        
        self.compare_parameters(initial_state)

        return current_loss, current_accuracy
        #Do not forget to include the scheduler.step() in the run funciton. 

    
    def save_current_parameters(self):
        print(f"{Fore.LIGHTMAGENTA_EX} {[name for name, param in self.model.named_parameters()]}")
        initial_model = {} 
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                initial_model[name] = param.data.clone()
                initial_model[name + '_grad'] = param.grad.clone()
                print(f"{Fore.LIGHTRED_EX} The shape of the parameter {name} is {param.grad.shape} and required grad is {param.requires_grad}")

        return initial_model


    def compare_parameters(self, initial_model):
        for current_name, current_param in self.model.named_parameters():
            if current_name not in initial_model:
                continue

            initial_name = current_name
            initial_param = initial_model[initial_name]
            initial_grad = initial_model[initial_name + '_grad']

            if torch.allclose(initial_param, current_param):
                print(f"{Fore.LIGHTRED_EX} The parameters are the same for the parameter name {initial_name}")

            if torch.allclose(initial_grad, current_param.grad):
                print(f"{Fore.LIGHTRED_EX} The gradients are the same for the parameter name {initial_name}")
            
            print(f"{Fore.MAGENTA} Params : {initial_name} --> Initials {initial_param.data.norm()}  || Current {current_param.data.norm()} ")
            print(f"{Fore.MAGENTA} Grads : {initial_name} --> Initials {initial_grad.norm()}  || Current {current_param.grad.norm()} ")
            

    
    def validate_epoch(self):
        self.model.eval()
        current_loss = 0.0
        current_accuracy = 0.0

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_loader), desc='Validation'):
                images, labels = self.to_device(batch)
                pred = self.model(images)
                loss = self.loss(pred, labels)
                print(Fore.LIGHTRED_EX + "[LOSS INFO {}th iteration] The validation loss is {:.3f}".format(idx, loss.item()) + Style.RESET_ALL)
                print(Fore.BLUE + "The predicitons are {}".format(pred.argmax(1)) + Fore.GREEN + "\n The true labels are {}".format(labels))
                current_loss += loss.item()
                current_accuracy += (pred.argmax(1) == labels).float().mean().item()
                print(Fore.LIGHTRED_EX + "[ACCURACY INFO {}th iteration] The validation accuracy is {:.3f}".format(idx, (pred.argmax(1) == labels).float().mean().item()) + Style.RESET_ALL)

            current_loss /= len(self.test_loader)
            current_accuracy /= len(self.test_loader)
            
            self.val_accuracy.append(current_accuracy)
            self.val_loss.append(current_loss)

            return current_loss, current_accuracy
        
        
    def save_best_model(self, current_loss, current_accuracy):
        if current_loss < self.best_loss:
            self.not_improved = 0
            self.best_loss = current_loss
            self.best_accuracy = current_accuracy
            save_path = "AlexNet/best_model.pth"

            print("\033[97m [SAVING INFO] Saving the best model with loss: {:.3f} and accuracy: {:.3f} \033[0m]".format(self.best_loss, self.best_accuracy))
            torch.save(
                {
                    'epoch' : self.epoch,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'loss' : current_loss,
                    'accuracy' : current_accuracy
                }, save_path
            )

        else:
            self.not_improved += 1
            print("\033[91m [PATIENCE INFO] The model has not been improved for {} epochs \033[0m".format(self.not_improved))
            if self.not_improved >= self.patience:
                print("\033[91m [STOP INFO] Stopping the trainining after {} epochs\n Saving the final model \033[0m".format(self.patience))
                self.save_final_model()
                exit()
    
    
    def save_final_model(self):
        print("\033[98m [SAVING INFO] Saving the final model")
        save_path = "AlexNet/final_model.pth"
        torch.save(
            {
                'epoch' : self.epoch,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                'loss' : self.best_loss,
                'accuracy' : self.best_accuracy
            }, save_path
        )


    def verbose(self, epoch, train_loss, train_acc, val_loss, val_acc):
        print("\033[95m ==> Epoch {}: Training Accuracy --> {:.3f} || Training Loss --> {:.3f} \033[0m".format(epoch, train_acc, train_loss))
        print("\033[95m ==> Epoch {}: Validation Accuracy --> {:.3f} || Validation Loss --> {:.3f} \033[0m".format(epoch, val_acc, val_loss))
        print("\033[95m ==> Best Accuracy: {:.3f} || Best Loss: {:.3f} \033[0m\n".format(self.best_accuracy, self.best_loss))

    
    def run(self):
        print("\033[92m [INFO] Starting the tranining process \033[0m ")
        for epoch in range(1, self.epoch + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            self.scheduler.step(val_loss)
            self.save_best_model(val_loss, val_acc)
            self.verbose(epoch, train_loss, train_acc, val_loss, val_acc)

        print("\033[92m [INFO] Training has been completed \033[0m")
        self.save_final_model()
        return self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy
    

    def __call__(self):
        return self.run()


if __name__ == '__main__':
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
    train_model = TrainModel(data_path)
    train_model()
