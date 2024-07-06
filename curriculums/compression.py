"""Compression-related metrics."""

from numpy  import asarray, mean, ndarray, sqrt
from PIL    import Image

def rmse(image: ndarray) -> float:
    """# Calculates the root mean squared error between an image and it's compressed/resized form.

    ## Args:
        * image (array):    Image input.

    ## Returns:
        * float:    Root mean squared error.
    """
    # Record image shape
    shape:  tuple = image.shape
    
    # Calculate & return RMSE
    return sqrt(mean(
        (image - asarray(
            Image.fromarray(image).resize((shape[1] // 2, shape[0] // 2)).resize((shape[1], shape[0]))
        )) ** 2
    ))