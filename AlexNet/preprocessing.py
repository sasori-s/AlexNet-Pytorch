import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
from fancypca import FancyPCA
from torch.utils.data import DataLoader, Dataset
from colorama import Fore, Style, init

init(autoreset=True)


class LoadLessDataset(datasets.ImageFolder):
    def __init__(self, root=None,  transform=None):
        super(LoadLessDataset, self).__init__(root=root, transform=transform)

    def find_classes(self, directory):
        limit = 5
        classes = []

        for d in os.scandir(directory):
            if d.is_dir():
                limit -= 1
                classes.append(d.name)
                if limit == 0:
                    break
        
        classes_to_idx = {name : i for i, name in enumerate(classes)}
        return classes, classes_to_idx


class LoadDataset():
    def __init__(self, train_path, test_path=None, batch_size=128, testing_mode=False):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.testing_mode = testing_mode


    def augment_dataset(self, single_image = False, single_image_path="/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals/antelope/0c16ef86c0.jpg"):
        train_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.RandomCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            FancyPCA(),
            ConvertToFloat(scale=True)
        ])

        test_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.FiveCrop(size=(224, 224)),  #[crop1, crop2, crop3, crop4, ...]
            RandomFlipOnFiveCrop(p=0.5),
            FancyPCA(),
            ConvertToFloat(scale=True),
        ])

        if not single_image:
            if self.testing_mode:
                train_dataset = LoadLessDataset(self.train_path, transform=train_transform)
                test_dataset = LoadLessDataset(self.test_path, transform=test_transform)

            else:
                train_dataset = datasets.ImageFolder(self.train_path, transform=train_transform)
                test_dataset = datasets.ImageFolder(self.test_path, transform=test_transform)

            test_dataset = MyDataSet(test_dataset)
            print("\033[92m", len(train_dataset), "\033[0m")
            print("\033[92m", len(test_dataset), "\033[0m")
            # print(f"{Fore.GREEN} {test_dataset.classes}")

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

            # self._show_images(test_dataset=test_dataset)
            # self._show_images(train_dataset=train_dataset)
            return self.train_loader, self.test_loader
        
        else:
            image = Image.open(single_image_path)
            self.single_train_augmented = train_transform(image)
            self.single_test_augmented = test_transform(image)
            self._show_images(test_dataset=self.single_test_augmented, single_image=False)
            self._show_images(train_dataset=self.single_train_augmented, single_image=False)

            return self.single_train_augmented, None

        
    def reshape_dataset(self, dataset):
        new_dataset = []
        target = torch.empty(3, dtype=torch.long).random_(5)
        for i in range(len(dataset)):
            pass


    def _show_images(self, train_dataset=None, test_dataset=None, single_image=False):

        if not single_image:
            figure = plt.figure(figsize=(8, 8))
            cols, rows = 3, 3

            if test_dataset is not None:
                for i in range(9):
                    sample_index = torch.randint(len(self.test_loader.dataset), size=(1,)).item()
                    image, label = test_dataset[sample_index]
                    # print(f"The shape of the image is {image.shape}" )
                    figure.add_subplot(rows, cols, i + 1)
                    plt.title(test_dataset.data.classes[label])
                    # plt.title(label)
                    plt.axis('off')
                    plt.imshow(np.asarray(image.permute(1, 2, 0)).squeeze())
                
                plt.show()
            else:
                for i in range(9):
                    sample_index = torch.randint(len(self.train_loader.dataset), size=(1,)).item()
                    image, label = train_dataset[sample_index]
                    # print(f"The shape of the image is {image.shape}" )
                    figure.add_subplot(rows, cols, i + 1)
                    plt.title(train_dataset.classes[label])
                    plt.axis('off')
                    plt.imshow(np.asarray(image.permute(1, 2, 0)).squeeze())
                
                plt.show()

        else:
            # print("The type of the image is ", train_dataset[0].shape)

            if train_dataset.shape == (5, 3, 224, 224):
                figure, axes = plt.subplots(1, 5, figsize=(15, 10))
                for i in range(5):
                    axes[i].imshow(np.asarray(train_dataset[i].permute(1, 2, 0)).squeeze())
                    axes[i].axis('off')
                plt.show()
            else:
                plt.imshow(np.asarray(train_dataset.permute(1, 2, 0)).squeeze())
                plt.show()


class MyDataSet(Dataset):
    def __init__(self, data):
        super(MyDataSet, self).__init__()
        self.data = data
        self.classes = data.classes
    
    def __len__(self):
        return len(self.data) * 5
    
    def __getitem__(self, index):
        original_image_index = index // 5
        crop_index = index % 5
        stack_image_tensor, label = self.data[original_image_index]
        image_tensor = stack_image_tensor[crop_index]
        return image_tensor, label


class RandomFlipOnFiveCrop:
    def __init__(self, p=0.5):
        self.p = p
        self.flip = v2.RandomHorizontalFlip(p=self.p)

    
    def __call__(self, crops):
        return [self.flip(crop) for crop in crops]


class ConvertToFloat:
    def __init__(self, scale=True):
        self.scale = scale
        self.convertion = v2.ToTensor()
        self.dtype_conversion = lambda x : x.to(torch.float32)

    def __call__(self, images):
        if isinstance(images, Image.Image):
            return (self.dtype_conversion(self.convertion(images)))

        list_to_return =  torch.stack([self.dtype_conversion(self.convertion(image)) for image in images], dim=0)
        # print("\033[91mThe shape of the list_to_return is ", list_to_return.shape, "\033[0m")
        return list_to_return


if __name__ == '__main__':
    data_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals'
    load_dataset = LoadDataset(data_path, data_path, testing_mode=True)
    train_loader, test_loader = load_dataset.augment_dataset(single_image=False)
    
    it = iter(train_loader)
    image, label = next(it)
    print(image.shape)
    print(label)
