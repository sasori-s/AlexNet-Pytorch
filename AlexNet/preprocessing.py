import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image

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


class FancyPCA:
    def __init__(self, image_path=None):
        self.image_path = image_path


    def __call__(self, image):
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            # print("The type of the image is list")
            list_to_return = [self._calculate_eigens(img) for img in image]
            # print("The type of the list_to_return is ", len(list_to_return))
            return list_to_return
        else:
            return self._calculate_eigens(image)


    def _calculate_eigens(self, image):
        # image = Image.open(self.image_path)
        image_np = np.asarray(image)

        image_np = image_np / 255.0
        self.image_np = image_np.copy()

        # if isinstance(image, list) and isinstance(image[0], Image.Image):
        #     image_np = np.asarray(image[1])

        image_flatten = image_np.reshape(-1, 3)
        # print(f"Image at the starting of _calculate_eigen function: {image_flatten.shape}")
        image_centered = image_flatten - np.mean(image_flatten, axis=0)
        image_covariance = np.cov(image_centered, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(image_covariance)

        sorted_index = eigen_values[::-1].argsort()
        eigen_values[::-1]

        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors[:, sorted_index]
        return self._apply_pca()

    
    def _apply_pca(self,):
        augmented_image = self.image_np.copy()
        matrix1 = np.column_stack((self.eigen_vectors))
        matrix2 = np.zeros((3, 1))

        alpha = np.random.normal(0, 0.1)

        matrix2[:, 0] = alpha * self.eigen_values[:]
        added_matrix =  np.matrix(matrix1) * np.matrix(matrix2)
        # print(f"Added matrix: {added_matrix}")  
        for i in range(3):
            augmented_image[..., i] += added_matrix[i]
        
        augmented_image = np.clip(augmented_image, 0, 1)
        augmented_image = np.uint8(augmented_image * 255)

        # print(f"Shape of the original image: {augmented_image.shape}")
        # print(f"Shape of the augmented image: {self.image_np.shape}")

        # self._plot_images(self.image_np, augmented_image)
        final_image = Image.fromarray(augmented_image)
        return final_image



    def _plot_images(self, image1, image2):
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(image1)
        axes[1].imshow(image2)
        plt.suptitle('Original Image vs Augmented Image')
        plt.show()


if __name__ == '__main__':
    data_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals'
    load_dataset = LoadDataset(data_path, data_path)
    train_loader, test_loader = load_dataset.augment_dataset(single_image=False)