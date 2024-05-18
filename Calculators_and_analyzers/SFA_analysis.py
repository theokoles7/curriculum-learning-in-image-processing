import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This file uses FFT to analyze an image

# Replace image as needed
image = Image.open("Cats_color/Noise_reduction.png")

# Image is loaded and made greyscale
image_grey = np.asarray(image.convert("L"))
image_array = np.asarray(image)

image_red = image_array[:, :, 0]
image_green = image_array[:, :, 1]
image_blue = image_array[:, :, 2]

# FFT on the image
freq_grey = np.abs(np.fft.fft2(image_grey))
freq_red = np.abs(np.fft.fft2(image_red))
freq_green = np.abs(np.fft.fft2(image_green))
freq_blue = np.abs(np.fft.fft2(image_blue))

# Plotting
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))

image_frequency_array = [
    [image_grey, freq_grey, "Grey"],
    [image_red, freq_red, "Red"],
    [image_blue, freq_blue, "Blue"],
    [image_green, freq_green, "Green"],
]

for i, img in enumerate(image_frequency_array):
    # Image Frequencies
    ax[i, 0].imshow(img[0], interpolation="none")
    ax[i, 0].set_title(f"Image ({img[2]})")

    log_freq = np.log(img[1])

    stdev = np.std(log_freq).round(2)
    mean = np.mean(log_freq).round(2)

    # Logarithmic Analysis Histogram
    # clipped to prevent division by zero errors
    clipped_frequency = np.clip(log_freq.ravel(), 0, np.inf)
    ax[i, 1].hist(clipped_frequency, bins=100)
    ax[i, 1].set_title(f"Logarithmic Frequency Histogram ({img[2]})")

    # Statistics
    ax[i, 2].axis("off")
    ax[i, 2].text(
        0.2,
        0.7,
        f"st.dev. = {stdev} \nmean = {mean} ",
        bbox={"facecolor": "wheat", "alpha": 0.5, "pad": 10},
    )

# Making it look nice
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.3)
plt.show()
