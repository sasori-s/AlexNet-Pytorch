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

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class LoadLessDataset(datasets.ImageFolder):
    def __init__(self, root=None,  transform=None, num_classes=10):
        self.num_classes = num_classes
        super(LoadLessDataset, self).__init__(root=root, transform=transform)
        

    def find_classes(self, directory):
        limit = self.num_classes
        classes = []

        for d in os.scandir(directory):
            if d.is_dir():
                limit -= 1
                classes.append(d.name)
                if limit == 0:
                    break
        
        classes_to_idx = {name : i for i, name in enumerate(classes)}
        return classes, classes_to_idx


#this class is for test.py
class LoadTestData(datasets.ImageFolder):
    def __init__(self, root, test_classes, transform=None):
        self.test_classes = test_classes
        super(LoadTestData, self).__init__(root=root, transform=transform)

    
    def find_classes(self, directory):
        classes = self.test_classes
        classes_to_idx = {name : i for i, name in enumerate(classes)}
        return classes, classes_to_idx


class LoadDataset():
    def __init__(self, train_path, test_path=None, batch_size=128, testing_mode=False, num_classes=90, with_gpu=False, is_test=False, test_classes=None):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.testing_mode = testing_mode
        self.num_classes = num_classes
        self.with_gpu = with_gpu
        self.is_test = is_test
        self.test_classes = test_classes


    def augment_dataset(self):
        train_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            FancyPCA(),
            v2.RandomCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            ConvertToFloat(scale=True)
        ])

        test_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            FancyPCA(),
            v2.FiveCrop(size=(224, 224)),  #[crop1, crop2, crop3, crop4, ...]
            RandomFlipOnFiveCrop(p=0.5),
            ConvertToFloat(scale=True),
        ])

        if self.testing_mode:
            train_dataset = LoadLessDataset(self.train_path, transform=train_transform, num_classes=self.num_classes)
            test_dataset = LoadLessDataset(self.test_path, transform=test_transform, num_classes=self.num_classes)
        
        #This is for test.py
        if self.is_test:
            test_dataset = LoadTestData(self.train_path, self.test_classes, transform=test_transform)
            test_dataset = MyDataSet(test_dataset)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)
            return self.test_loader

        else:
            train_dataset = datasets.ImageFolder(self.train_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(self.test_path, transform=test_transform)

        test_dataset = MyDataSet(test_dataset)
        print("\033[92m", len(train_dataset), "\033[0m")
        print("\033[92m", len(test_dataset), "\033[0m")
        
        if not self.with_gpu:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

            return self.train_loader, self.test_loader

        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

            train_prefetcher = CUDAPrefetcher(train_loader, device)
            test_prefetcher = CUDAPrefetcher(test_loader, device)

            return train_prefetcher, test_prefetcher


    def no_augment_dataset(self):
        train_transform = v2.Compose([
            v2.Resize(size=(224, 224)),
            v2.ToTensor()
        ])

        train_dataset = datasets.ImageFolder(self.train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(self.test_path, transform=train_transform)

        print("\033[92m", f"Train dataset length -> {len(train_dataset)}", "\033[0m")
        print("\033[96m", f"validation dataset length -> {len(test_dataset)}" "\033[0m")

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        return self.train_loader, self.test_loader

        
    def reshape_dataset(self, dataset):
        new_dataset = []
        target = torch.empty(3, dtype=torch.long).random_(5)
        for i in range(len(dataset)):
            pass


    def _show_images(self, train_dataset=None, test_dataset=None, single_image=False):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3

        if test_dataset is not None:
            for i in range(9):
                sample_index = torch.randint(len(self.test_loader.dataset), size=(1,)).item()
                image, label = test_dataset[sample_index]
                figure.add_subplot(rows, cols, i + 1)
                plt.title(test_dataset.data.classes[label])
                plt.axis('off')
                plt.imshow(np.asarray(image.permute(1, 2, 0)).squeeze())
            
            plt.show()
        else:
            for i in range(9):
                sample_index = torch.randint(len(self.train_loader.dataset), size=(1,)).item()
                image, label = train_dataset[sample_index]
                figure.add_subplot(rows, cols, i + 1)
                plt.title(train_dataset.classes[label])
                plt.axis('off')
                plt.imshow(np.asarray(image.permute(1, 2, 0)).squeeze())
            
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


#This class was taken from https://github.com/Lornatang/AlexNet-PyTorch/blob/main/dataset.py and modified to fit my needs
class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            # print(self.batch_data[1])
            for i, tensor_data_label in enumerate(self.batch_data):
                if torch.is_tensor(tensor_data_label[0]):
                    self.batch_data[0] = self.batch_data[0].to(self.device, non_blocking=True)
                if torch.is_tensor(tensor_data_label[1]):
                    self.batch_data[1] = self.batch_data[1].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


if __name__ == '__main__':
    data_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals'
    load_dataset = LoadDataset(data_path, data_path, testing_mode=True, num_classes=10)
    train_loader, test_loader = load_dataset.augment_dataset()
    
    it = iter(train_loader)
    image, label = next(it)

    print(image.shape)
    print(label.device)

    train_gpu_fethcer = CUDAPrefetcher(train_loader, device)
    train_data = train_gpu_fethcer.next()
    print(train_data[0].device)
    print(train_data[1].device)
