import torch.nn as nn
import os
import torch
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import numpy as np
init(autoreset=True)

class TrainingUtils():
    def __init__(self):
        data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
        self.plot_save_path = "AlexNet/training_plots"
        
    
    @staticmethod
    def backward_hook(model_layer, grad_input, grad_output):
        print(f"{Fore.LIGHTGREEN_EX} The layer is {model_layer} \t{Fore.BLUE} Input has {len(grad_input)} elements \t{Fore.CYAN} Output has {len(grad_output)} elements")

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
    

    def call_backward_hook(self, model : object):
        # layers_to_track = [nn.modules.conv.Conv2d, nn.modules.linear.Linear, nn.modules.pooling.MaxPool2d, nn.modules.activation.ReLU, nn.modules.dropout.Dropout, nn.modules.loss.CrossEntropyLoss]
        layers_to_track = [nn.modules.linear.Linear]

        print(f"{Fore.LIGHTWHITE_EX} {model.modules()}")
        for layer in model.modules():
            print(f"{Fore.LIGHTRED_EX} The layer is {type(layer)}")
            if isinstance(layer, tuple(layers_to_track)):
                layer.register_full_backward_hook(self.backward_hook)


    def save_current_parameters(self, model : object):
        # print(f"{Fore.LIGHTMAGENTA_EX} {[name for name, param in model.named_parameters()]}")
        initial_model = {} 
        for name, param in model.named_parameters():
            if param.grad is not None:
                initial_model[name] = param.data.clone()
                initial_model[name + '_grad'] = param.grad.clone()
                print(f"{Fore.LIGHTRED_EX} The shape of the parameter {name} is {param.grad.shape} and required grad is {param.requires_grad}")

        return initial_model
    
    
    def compare_parameters(self, model : object, initial_model):
        for current_name, current_param in model.named_parameters():
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


    def save_best_model(self, model : object, current_loss, current_accuracy):
        if current_loss < model.best_loss:
            model.not_improved = 0
            model.best_loss = current_loss
            model.best_accuracy = current_accuracy
            save_path = "AlexNet/best_model.pth"

            print("\033[97m [SAVING INFO] Saving the best model with loss: {:.3f} and accuracy: {:.3f} \033[0m]".format(model.best_loss, model.best_accuracy))
            torch.save(
                {
                    'epoch' : model.epoch,
                    'model_state_dict' : model.model.state_dict(),
                    'optimizer_state_dict' : model.optimizer.state_dict(),
                    'loss' : current_loss,
                    'accuracy' : current_accuracy
                }, save_path
            )

        else:
            model.not_improved += 1
            print("\033[91m [PATIENCE INFO] The model has not been improved for {} epochs \033[0m".format(model.not_improved))
            if model.not_improved >= model.patience:
                print("\033[91m [STOP INFO] Stopping the trainining after {} epochs\n Saving the final model \033[0m".format(model.patience))
                self.save_final_model(model)
                exit()
    
    
    def save_final_model(self, model : object):
        print("\033[98m [SAVING INFO] Saving the final model")
        save_path = "AlexNet/final_model.pth"
        torch.save(
            {
                'epoch' : model.epoch,
                'model_state_dict' : model.model.state_dict(),
                'optimizer_state_dict' : model.optimizer.state_dict(),
                'loss' : model.best_loss,
                'accuracy' : model.best_accuracy
            }, save_path
        )


    def verbose(self, model : object,  epoch, train_loss, train_acc, val_loss, val_acc):
        print("\033[95m ==> Epoch {}: Training Accuracy --> {:.3f} || Training Loss --> {:.3f} \033[0m".format(epoch, train_acc, train_loss))
        print("\033[95m ==> Epoch {}: Validation Accuracy --> {:.3f} || Validation Loss --> {:.3f} \033[0m".format(epoch, val_acc, val_loss))
        print("\033[95m ==> Best Accuracy: {:.3f} || Best Loss: {:.3f} \033[0m\n".format(model.best_accuracy, model.best_loss))

    
    def see_grad(self, model : object):
        for name, reLU in model.activations.items():
            handle = model.activations[name].register_hook(
                lambda grad : print(f"{Fore.LIGHTRED_EX} The grad of {name} is {grad.sum()}")
            )
            handle.remove()


    def verbose(self, model, epoch, train_loss, train_acc, val_loss, val_acc):
        print("\033[95m ==> Epoch {}: Training Accuracy --> {:.3f} || Training Loss --> {:.3f} \033[0m".format(epoch, train_acc, train_loss))
        print("\033[95m ==> Epoch {}: Validation Accuracy --> {:.3f} || Validation Loss --> {:.3f} \033[0m".format(epoch, val_acc, val_loss))
        print("\033[95m ==> Best Accuracy: {:.3f} || Best Loss: {:.3f} \033[0m\n".format(model.best_accuracy, model.best_loss))

    
    def plot_grad_flow(self, model, named_parameters, epoch, idx):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                max_grads.append(p.grad.abs().max().cpu().detach().numpy())
        plt.figure(figsize=(10, 10))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="green")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="red")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.tight_layout()
        plot_name = os.path.join(self.plot_save_path, f"gradient_flow_{epoch}_{idx}.png")
        plt.savefig(plot_name)
        # plt.legend([Line2D([0], [0], color="c", lw=4),
        #             Line2D([0], [0], color="b", lw=4),
        #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])