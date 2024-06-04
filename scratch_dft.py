from PIL    import Image
import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image = Image.open("images/smiley_face.jpeg")

transform = transforms.Compose([
    transforms.PILToTensor()
])

image_tensor = transform(image)

image_transform = fft.fftn(image_tensor)

print(image_transform)

print(f"Shape: {image_transform.shape}")

image_array = np.array(np.squeeze(image_transform).permute(1, 2, 0), dtype = float)

plt.imshow(image_array)

plt.show()