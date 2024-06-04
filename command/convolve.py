"""Convolution operations."""

from logging    import Logger
from numpy      import array
from torch      import Conv2d

from utils      import LOGGER

# Initialize logger
_logger:    Logger =    LOGGER.getChild("convolution")

def convolve(image: array, kernel: array) -> array:
    """# Perform convolution on an image.

    ## Args:
        * image   (array): Image being convolved upon.
        * kernel  (array): Kernel used in convolution.

    ## Returns:
        * array: Convolved image output.
    """