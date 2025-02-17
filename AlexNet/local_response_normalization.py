import torch
import torch.nn.functional as F

class LocalResponseNormalization:
    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def calculate_normalization(self, a):  #[a1, a2, a3, a4, a5]
        self.output = torch.zeros(a.size())

        for i in range(a.size()):
            squared_sum = [a[j] ** 2 for j in range(max(0, i - self.n // 2), min(len(a) - 1, i + self.n // 2))]
            normalized_denominator = torch.pow(self.k + self.alpha * sum(squared_sum), self.beta)
            normalized_activation = a[i] / normalized_denominator
            self.output[i] = normalized_activation
        
        return self.output
    
    
    def forward(self, a):
        """
        Applies local response normalization (LRN) to the input tensor.

        Local response normalization is commonly used in neural networks to normalize 
        the activations across channels, enhancing generalization and reducing sensitivity 
        to large activations in neighboring channels.

        Args:
            x (torch.Tensor): Input tensor of shape (W, H, C), where:
                - W: Width of the image or feature map
                - H: Height of the image or feature map
                - C: Number of channels (e.g., 3 for RGB images)

        Returns:
            torch.Tensor: Normalized tensor of the same shape (W, H, C).

        Normalization is performed using the formula:
        
            x_normalized = x / (k + alpha * sum(x^2 over neighboring channels))^beta
        
        where:
            - `n` is the number of neighboring channels to consider for normalization.
            - `k` is a small constant to avoid division by zero.
            - `alpha` and `beta` are scaling parameters.

        Steps:
            1. Compute squared activations of the input tensor.
            2. Apply 1D convolution along the channel dimension to compute the sum of squared activations 
            over neighboring channels.
            3. Compute the normalization denominator using the given formula.
            4. Normalize the input tensor by dividing it by the computed denominator.

        Note:
            - The padding mode is set to "replicate" to extend border values.
            - The convolution is performed in 1D across the channel dimension.
        """
        if a.dim() == 1:
            a = a.view(1, 1, -1)

        elif a.dim() == 2:
            a = a.unsqueeze(0)
        
        elif a.dim() == 3:
            pass

        else:
            raise ValueError("Input tensor must be 1D, 2D, or 3D")

        C, W, H = a.size()

        squared_a = torch.pow(a, 2)
        print(f"\t Squeezed input shape : {squared_a.squeeze().shape}")
        kernel = torch.ones(1, W, self.n)
        # kernel = torch.ones(1, self.n, 1, 1)


        print(f"\t Kernel shape : {kernel.shape}")
        squared_a_padded = F.pad(squared_a, (self.n // 2, self.n // 2), mode='constant', value=0)
        # squared_a_padded = F.pad(squared_a, (0, 0, 0, 0, self.n // 2, self.n // 2), mode='constant', value=0)


        print(f"\t Squeezed padded input shape : {squared_a_padded.squeeze().shape}")
        squared_sum = F.conv1d(squared_a_padded.squeeze(), kernel, stride=1, padding=0)
        # squared_sum = F.conv2d(squared_a_padded.unsqueeze(0), kernel.unsqueeze(0), stride=1, padding=0, groups=1)
        denominator = torch.pow(self.k + self.alpha * squared_sum, self.beta)
        print(f"\t Denominator shape : {denominator.shape}")
        normalized_a = a / denominator

        return normalized_a.view_as(a)
        

    def forward2(self, a):
        # Ensure the input tensor has the correct shape (C, H, W)
        if a.dim() == 1:
            a = a.view(1, 1, -1)  # Reshape to (1, 1, L) for 1D input
        elif a.dim() == 2:
            a = a.unsqueeze(0)  # Reshape to (1, H, W) for 2D input
        elif a.dim() == 3:
            pass  # Input is already in (C, H, W) format
        else:
            raise ValueError("Input tensor must be 1D, 2D, or 3D.")

        C, H, W = a.size()

        squared_a = torch.pow(a, 2)
        print(f"\t Squared input shape: {squared_a.unsqueeze(0).shape}")
        kernel = torch.ones(C, (C  + self.n // 2) // C, W, H)
        # kernel = torch.ones(1, self,n, 1, 1)
        squared_a_padded = F.pad(squared_a.unsqueeze(0), (0, 0, 0, 0, self.n // 2, self.n//2), mode='constant', value=0)
        print(f"\t Squared padded input shape: {squared_a_padded.squeeze().shape}")
        squared_sum = F.conv2d(squared_a.unsqueeze(0), kernel, stride=1, padding=0, groups=3)
        print(f"\t Squared sum shape: {squared_sum.squeeze().shape}")
        denominator = torch.pow(self.k + self.alpha * squared_sum, self.beta)
        print(f"\t Denominator shape: {denominator.squeeze().shape}")
        normalized_a = a / denominator
        print(f"\t Normalized shape: {normalized_a.squeeze().shape}")
        return normalized_a.view_as(a)


    def forward3(self, a):
        device = a.device
        print("\tTHe device is {}".format(device))
        if a.dim() == 3:  # (C, H, W)
            a = a.unsqueeze(0)  # Add batch dimension, now (1, C, H, W)
        elif a.dim() != 4:
            raise ValueError("Input must be a 3D (C, H, W) or 4D (N, C, H, W) tensor.")

        N, C, H, W = a.size()
        
        squared_a = a ** 2

        # Apply 1D convolution across the channel dimension
        # Kernel size is (1, n, 1, 1) to operate on n neighboring channels at each spatial position
        kernel = torch.ones(1, C + self.n - 1, 1, 1, device=a.device)  # Shape: (1, n + c -1, 1, 1)
        print(f"\t Kernel shape: {kernel.shape}")
        
        # Pad the input along the channel dimension (n // 2 on both sides)
        padding = self.n // 2
        squared_a_padded = F.pad(squared_a, (0, 0, 0, 0, padding, padding), mode='constant', value=0).to(device)
        print(f"\t Squared padded input shape: {squared_a_padded.squeeze().shape}")
        print("\t The Squared_a_padded resides in the device: ", squared_a_padded.device)

        # Perform convolution to get the sum of squared values in the local channel window
        squared_sum = F.conv2d(squared_a_padded, kernel, stride=1, padding=0, groups=1)
        print(f"\t Squared sum shape: {squared_sum.squeeze().shape}")
        print("\t The Squared_sum resides in the device: ", squared_sum.device)

        # Compute the normalization denominator
        denominator = torch.pow(self.k + (self.alpha / self.n) * squared_sum, self.beta).to(device)
        print(f"\t Denominator shape: {denominator.squeeze().shape}")
        print("\t The denominator resides in the device: ", denominator.device)

        # Normalize the input
        normalized_a = a / denominator
        print(f"\t Normalized shape: {normalized_a.squeeze().shape}")
        print("\t The normalized_a resides in the device: ", normalized_a.device)

        return normalized_a.view_as(a)

    
    def __call__(self, a):
        # return torch.stack([self.forward2(a[i]) for i in range(a.size(0))])
        return self.forward3(a)
    

def torch_lrn(a, k, alpha, beta, n):
    lrn = torch.nn.LocalResponseNorm(size=n, alpha=alpha, beta=beta, k=k)
    output = lrn(a)
    return output


if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 224, 224)
    k = 2
    beta = 0.75
    alpha = 1e-4
    n = 5

    lrn = LocalResponseNormalization(k, n, alpha, beta)
    output = lrn(input_tensor)
    print("\t", output.shape)
    print(output[0][0][0][100])

    verify_output = torch_lrn(input_tensor, k, alpha, beta, n)
    print("\t", verify_output.shape)
    print(verify_output[0][0][0][100])


    print(torch.allclose(output, verify_output))