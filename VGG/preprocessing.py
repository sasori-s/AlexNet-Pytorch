import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class Preprocess(datasets.ImageFolder):
    def __init__(
            self, 
            data_path : str, 
            image_size : int
    ) -> None: 
        super(Preprocess, self).__init__(data_path)
        self.data_path = data_path
        self.image_size = image_size
        self.load_reshaped_image()

    
    def load_reshaped_image(
            self,
    ) -> Image:
        image = Image.open(self.data_path)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        return image
    

if __name__ == '__main__':
    preproces = Preprocess("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train/apple/apple_s_000027.png", 256)
