import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image

import pywt
import pywt.data as data

# print(pywt.wavelist(kind="discrete"))

def calculate_wavelet_energy(image, wavelet='db1', levels=1):
    print(f"Image size: {image.shape}")
    
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    
    # print(f"Coefficients: {coeffs}")
    
    print(f"Coefficient length: {len(coeffs)}")
    print(f"Coeffs[0]: {coeffs[0]}")
    print(f"Coeffs[1]: {coeffs[1]}")
    
    total_energy = 0
    for level in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level]
        energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
        total_energy += energy
    return total_energy

for image in glob("generated_images/*.jpeg"):
    
    print(f"Energy of {image}: {calculate_wavelet_energy(np.asarray(Image.open(image)))}")

# # Load image
# original = data.camera()

# print(f"Image shape: {original.shape}")

# print(f"Energy: {calculate_wavelet_energy(original)}")

# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(original, 'bior1.3')

# print(f"Coefficients: {coeffs2}")

# LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()