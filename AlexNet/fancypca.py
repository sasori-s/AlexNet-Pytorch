from PIL import Image
import numpy as np

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