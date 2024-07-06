"""Spatial frequency metrics."""

from numpy      import abs, asarray, ndarray, sum
from numpy.fft  import fft2
from PIL        import Image

def spatial_frequency(image: ndarray) -> float:
    """# Calculate the spatial frequency of an image.

    ## Args:
        * image (ndarray):  Image input.

    ## Returns:
        * float:    Spatial frequency of image channels (gray, red, green, blue).
    """
    # Calculate Fourier transform of each channel
    return sum([
        abs(fft2(asarray(Image.fromarray(image).convert("L")))),
        abs(fft2(image[:, :, 0])),
        abs(fft2(image[:, :, 1])),
        abs(fft2(image[:, :, 2]))
    ])