import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
from fancypca import FancyPCA

class LoadDataset:
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path


    def augment_dataset(self, single_image = False, single_image_path="/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals/antelope/0c16ef86c0.jpg"):
        train_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.RandomCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            FancyPCA(),
            # v2.ToDtype(torch.float32, scale=True)
            ConvertToFloat(scale=True)
        ])

        test_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.FiveCrop(size=(224, 224)),  #[crop1, crop2, crop3, crop4, ...]
            RandomFlipOnFiveCrop(p=0.5),
            FancyPCA(),
            ConvertToFloat(scale=True),
            # v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(v2.ToTensor(crop)) for crop in crops]))
            # v2.ToDtype(torch.float32, scale=True),
            # v2.Lambda(lambda crops: torch.stack([v2.ToDtype(torch.float32, scale=True)(crop) for crop in crops])) 
        ])

        if not single_image:
            train_dataset = datasets.ImageFolder(self.train_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(self.test_path, transform=test_transform)
            print("\033[92m", type(train_dataset[0][0]), "\033[0m")
            print("\033[92m", type(test_dataset[0][0]), "\033[0m")

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

            self._show_images(test_dataset=test_dataset)
            self._show_images(train_dataset=train_dataset)
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
                    plt.title(test_dataset.classes[label])
                    plt.axis('off')
                    plt.imshow(np.asarray(image[2].permute(1, 2, 0)).squeeze())
                
                plt.show()
            else:
                for i in range(9):
                    sample_index = torch.randint(len(self.test_loader.dataset), size=(1,)).item()
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
    load_dataset = LoadDataset(data_path, data_path)
    train_loader, test_loader = load_dataset.augment_dataset(single_image=False)