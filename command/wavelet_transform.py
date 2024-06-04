"""Wavelet transform operations."""

from logging    import Logger
from numpy      import array, clip, asarray, ndarray, sum, uint8, zeros
from PIL        import Image

from utils      import LOGGER

# Initialize logger
_logger:    Logger =    LOGGER.getChild("transform")

def transform(image: str, wavelet: array) -> array:
    """# Perform wavelet transform on image.

    ## Args:
        * image     (array):    Image being transformed.
        * wavelet   (array):    Wavelet being used in transform.

    ## Returns:
        * array: Transformed image output.
    """
    # Open image and record shape
    img =                           asarray(Image.open(image))
    height, width, channels =       img.shape[0], img.shape[1]
    _logger.debug(f"Image shape(height: {height}, width: {width})")
    
    # Record kernel dimensions
    kernel_height, kernel_width =   wavelet.shape[0], wavelet.shape[1]
    _logger.debug(f"Wavelet shape(height: {kernel_height}, width: {kernel_width})")
    
    # Initialize new image of zeros
    new_img =                       zeros((height - kernel_height + 1, width - kernel_width + 1, channels))
    
    for row in range(kernel_height // 2, (kernel_width // 2) - 1):
        for col in range(kernel_height // 2, (kernel_width // 2) - 1):
            
            # Extract a window of pizels around the current pixel
            window = img[row - kernel_height // 2 : row + kernel_height // 2 + 1, col - kernel_width // 2 : col + kernel_width // 2 + 1]
            
            # Apply the wavlet to the window and set the results as the value of the current pizel in the new image
            new_img[row, col, 0] = int(sum(window[:, :, 0] * wavelet))
            new_img[row, col, 1] = int(sum(window[:, :, 1] * wavelet))
            new_img[row, col, 2] = int(sum(window[:, :, 2] * wavelet))
            
    # Clip values to the range 0-255
    new_img =                       clip(new_img, 0, 255)
    return new_img.astype(uint8)
    
def standrad_deviation(image: ndarray) -> float:
    """# Calculate standard deviation of image pixels.

    ## Args:
        * image   (array):    Image being analyzed.

    ## Returns:
        * float: Standard deviation of image.
    """