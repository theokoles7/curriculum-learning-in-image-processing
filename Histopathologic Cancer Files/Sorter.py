import csv
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pywt

# The name of the folder goes here
folder = "Solids_and_Checkerboards"
listdir = os.listdir(folder)

img_scores = np.array([[any for i in range(6)] for j in range(len(listdir) + 1)])
img_scores[0] = ["id", "RMSE", "Fourier", "WDH", "WDV", "WDD"]
# For every image int the folder

for i, image_path in tqdm(enumerate(listdir)):
    # Convert it to greyscale, array-ize
    i += 1
    try:
        img = Image.open(folder + "/" + image_path).convert("L")
    except:
        continue
    img_array = np.asarray(img)

    # Reshape the image
    img_shape = img_array.shape
    img_resize = img.resize((img_shape[1] // 2, img_shape[0] // 2)).resize(
        (img_shape[1], img_shape[0])
    )
    img_resize_array = np.asarray(img_resize)

    # RMSE
    # Get the RMSE between the original and the resized version
    img_rmse = np.sqrt(np.mean((img_array - img_resize_array) ** 2))
    img_scores[i, 0] = image_path.split(".")[0]
    img_scores[i, 1] = img_rmse

    # FOURIER
    # Calculate fourier complexity
    fourier = np.std(np.abs(np.fft.fft2(img_array)))
    img_scores[i, 2] = fourier

    # Wavelet Energies
    # Calculate Wavelet energies
    wd = pywt.wavedec2(img_array, "coif1", level=1)
    img_scores[i, 3] = np.sum(np.square(wd[1][0]))
    img_scores[i, 4] = np.sum(np.square(wd[1][1]))
    img_scores[i, 5] = np.sum(np.square(wd[1][2]))


with open("Solids_and_Checkerboards.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(img_scores)
