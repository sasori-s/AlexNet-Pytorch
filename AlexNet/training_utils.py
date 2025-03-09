import torch.nn as nn
import os
from train import TrainModel
from colorama import Fore, Style, init

init(autoreset=True)

class TrainingUtils(TrainModel):
    def __init__(self):
        data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
        super(TrainingUtils, self).__init__(data_path)
    
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
    

    def call_backward_hook(self):
        print(f"{Fore.LIGHTRED_EX} {self.model.modules()}")
        for layer in self.model.modules():
            # print(f"{Fore.LIGHTRED_EX} The layer is {layer}")
            layer.register_full_backward_hook(self.backward_hook)
