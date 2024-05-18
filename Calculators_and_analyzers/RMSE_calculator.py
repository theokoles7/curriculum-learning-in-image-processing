import numpy as np
from PIL import Image
import os

# Calculates the RMSE between an image and its compressed and resized form.


# The name of the folder goes here
folder = "random_images"

img_scores = [[], []]

# For every image int the folder
for image_path in os.listdir(folder):
    # Convert it to greyscale, array-ize
    img = Image.open(image_path).convert("L")
    img_array = np.asarray(img)

    # Reshape the image
    img_shape = img_array.shape
    img_resize = img.resize((int(img_shape[1] / 2), int(img_shape[0] / 2))).resize(
        (img_shape[1], img_shape[0])
    )
    img_resize_array = np.asarray(img_resize)

    # Get the RMSE between the original and the resized version
    img_rmse = np.sqrt(np.mean((img_array - img_resize_array) ** 2))
    img_scores[0].append(image_path)
    img_scores[1].append(img_rmse)

img_mean = np.mean(img_scores[1])
img_scores[1] = np.abs(np.subtract(img_scores[1], img_mean))

# Sort and print in order of complexity
for i in np.argsort(img_scores[1]):
    print(f"{img_scores[0][i]}:\t {img_scores[1][i]}")
