from model import Model
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF

class Trial(nn.Module):
    def __init__(self, image_folder_path):
        super(Trial, self).__init__()
        self.image_folder_path = image_folder_path
        self.model = Model()
        self.read_images()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    
    def read_images(self):
        self.pil_images = []
        for image in os.listdir(self.image_folder_path):
            image_path = os.path.join(self.image_folder_path, image)
            image = Image.open(image_path)
            image = image.resize((224, 224))
            self.pil_images.append(image)


    def __call__(self):
        self.model.to(self.device)
        label = 0

        for image in self.pil_images:
            image = TF.pil_to_tensor(image).to(torch.float).to(self.device)
            model_output = self.model(image)
            print("\033[92m[OUTPUT] The output of the model is {} \033[0m".format(model_output))
            


if __name__ == '__main__':
    image_folder_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals/antelope"
    trial_run = Trial(image_folder_path)
    trial_run()