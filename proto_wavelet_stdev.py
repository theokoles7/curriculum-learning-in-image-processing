from PIL import Image
import numpy as np

# This is an approximation of the mexican hat wavelet
# This multiplier increases the brightness of the output.
MULTIPLIER = 2

# [cen]ter, [edg]e, [cor]ner of the kernel.
# Note that the whole kernel's values add up to 0
CEN = 2.0*MULTIPLIER
EDG = -0.4*MULTIPLIER
COR = -0.1*MULTIPLIER

wd_kernel = np.array([
[COR, EDG, COR],
[EDG, CEN, EDG],
[COR, EDG, COR],
])

def wavelet_decomposition(img:np.array):
    """Passes the wavelet decomposition kernel (wd_kernel) over an image.

    Args:
        img (np.array): The input image.

    Returns:
        2D Int Array: A 2D array of pixel values, corresponding
            to the output image. 
            Convert it into an image with Image.fromarray()
    """
    # Get the height and width of the image
    height,width =img.shape[0],img.shape[1]

    # Get the height and width of the kernel
    kernel_height,kernel_width = wd_kernel.shape[0],wd_kernel.shape[1]

    # Create a new image of original img size minus the border 
    # where the convolution can't be applied
    new_img = np.zeros((height-kernel_height+1,width-kernel_width+1,3)) 

    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(kernel_height//2, height-kernel_height//2-1):
        for j in range(kernel_width//2, width-kernel_width//2-1):
            # Extract a window of pixels around the current pixel
            window = img[i-kernel_height//2 : i+kernel_height//2+1,
                         j-kernel_width//2 : j+kernel_width//2+1]

            # Apply the convolution to the window and set the result
            #   as the value of the current pixel in the new image
            new_img[i, j, 0] = int(np.sum(window[:,:,0] * wd_kernel))
            new_img[i, j, 1] = int(np.sum(window[:,:,1] * wd_kernel))
            new_img[i, j, 2] = int(np.sum(window[:,:,2] * wd_kernel))

    # Clip values to the range 0-255
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)

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

    new_img = np.zeros((height-3+1,width-3+1,3)) 

    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(3//2, height-3//2-1):
        for j in range(3//2, width-3//2-1):
            # Extract a window of pixels around the current pixel
            window = img[i-3//2 : i+3//2+1,j-3//2 : j+3//2+1]

            # Get the standard deviation of each 3x3 window
            new_img[i, j, 0] = int(np.std(window[:,:,0]))
            new_img[i, j, 1] = int(np.std(window[:,:,1]))
            new_img[i, j, 2] = int(np.std(window[:,:,2]))

    # Clip values to the range 0-255
    new_img = np.clip(new_img, 0, 255)
    # return the edited image and the sum of all standard deviations.
    return new_img.astype(np.uint8), new_img.sum()

# Load the image.
image = Image.open('dog3.jpg')

# Turn the image into an array and act upon it.
stdev_image, summation = standard_dev(wavelet_decomposition(np.asarray(image)))

# Print the multiplication of the image sizes
print(str(summation/(image.size[0]*image.size[1])))

# Create a PIL image from the new image and display it
sImg = Image.fromarray(stdev_image)
sImg.show()
