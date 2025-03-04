import torch.nn as nn
import os

class TrainingUtils(nn.Module):
    pass

def main():
    path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals'
    for directory in os.scandir(path)[10]:
        if directory.is_dir():
            print(directory.name)

if __name__ == '__main__':
    main()