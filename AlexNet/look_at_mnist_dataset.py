import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import time
from preprocessing import FancyPCA
from PIL import Image

class ExtracDataFromUbyte:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def extract_data(self):
        X, y = loadlocal_mnist(
            images_path=self.image_path,
            labels_path=self.label_path
        )

        return X / 255.0, y


class ShowImage:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def show_image(self, index):
        image = self.X[index].reshape(28, 28)
        label = self.y[index]
        print('Label:', label)
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # image_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    # label_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

    # data = ExtracDataFromUbyte(image_path, label_path)
    # X, y = data.extract_data()
    # print(X.shape, y.shape)
    # Image_ = ShowImage(X, y)
    # Image_.show_image(0)

    image_path = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals/antelope/0a37838e99.jpg'
    image = Image.open(image_path)
    fancy_pca = FancyPCA()
    augmented_image = fancy_pca(image)
    Image._show(augmented_image)

    pca = FancyPCA(image)


