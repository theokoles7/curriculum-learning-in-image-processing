import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image

import pywt
import pywt.data as data

# print(pywt.wavelist(kind="discrete"))


def calculate_wavelet_energy(image, wavelet="coif1", levels=1):
    print(f"Image size: {image.shape}")

    coeffs = pywt.wavedec2(image, wavelet, level=levels)

    # print(f"Coefficients: {coeffs}")

    # print(f"Coefficient length: {len(coeffs)}")
    # print(f"Coeffs[0]: {coeffs[0]}")
    # print(f"Coeffs[1]: {coeffs[1]}")

    total_energy = 0
    for level in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level]
        energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
        total_energy += energy
    return total_energy


for image in glob("Solids_and_Checkerboards/*.png"):

    print(
        f"Energy of {image}: {calculate_wavelet_energy(np.asarray(Image.open(image)))}"
    )
