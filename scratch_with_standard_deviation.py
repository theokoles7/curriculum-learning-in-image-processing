# A version of scratch.py with standard deviations

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import pywt

def standard_dev(img:np.array):
    """Passes a standard deviation kernel over the image

    Args:
        img (np.array): The input image

    Returns:
        2D Int Array: A 2D array of pixel values, corresponding
            to the output image. 
            Convert it into an image with Image.fromarray()
        Int: The sum of standard deviation values
    """
    # Get the height and width of the image
    height,width =img.shape[0],img.shape[1]

    new_img = np.zeros((height-2,width-2))

    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(1, height-3):
        for j in range(1, width-3):
            # Extract a window of pixels around the current pixel
            window = img[i-1 : i+2,j-1 : j+2]

            # Get the standard deviation of each 3x3 window
            stdev = np.std(window[:,:])
            new_img[i][j] = stdev

    # return the edited image and the sum of all standard deviations.
    return new_img.astype(np.float32), new_img.sum()

def plot_standard_deviation(layer:list, position:int, component:str):
    arr, sum = standard_dev(layer)
    arr = np.multiply(arr, 256)
    axes[position].imshow(Image.fromarray(arr), cmap=plt.cm.gray)
    axes[position].set_title(component + '\n' + str(sum))
    axes[position].set_axis_off()

x = pywt.data.camera().astype(np.float32)
shape = x.shape

level = 1       # What level of decomposition to draw

fig, axes = plt.subplots(1, 5, figsize=[14, 4])

# show the original image before decomposition
axes[0].set_axis_off()
axes[0].imshow(x, cmap=plt.cm.gray)
axes[0].set_title('Image')
axes[0].set_axis_off()

# compute the 2D DWT
c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)

# normalize each coefficient array independently for better visibility
c[0] /= np.abs(c[0]).max()
c[level] = [d/np.abs(d).max() for d in c[level]]


plot_standard_deviation(c[0], 1, "Approximation")
plot_standard_deviation(c[1][0], 2, "Horizontal")
plot_standard_deviation(c[1][1], 3, "Vertical")
plot_standard_deviation(c[1][2], 4, "Diagonal")

plt.suptitle("Note: Images have increased values for visualization purposes.")
plt.tight_layout()
plt.show()