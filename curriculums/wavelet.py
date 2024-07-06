"""Wavelet transform metrics."""

from warnings   import filterwarnings

import pywt

from numpy      import abs, array, finfo, hstack, log2, square, sum
from pywt       import wavedec2, wavelist

# Filter warnings
filterwarnings("ignore")

def wavelet_energy(
        image:      array, 
        wavelet:    str = "db2", 
        level:      int = 1,
        mode:       str = "periodization"
    ) -> int:
    """# Calculate the wavelet energy of an image.

    ## Args:
        * image     (array):            Image for which the wavelet energy will be calculated.
        * wavelet   (str, optional):    Wavelet used in transform. Defaults to 'db2'.
        * level     (int, optional):    Level(s) of decomposition to perform. Defaults to 1.
        * mode      (str, optional):    Mode of padding. Options: 'zero', 'constant', 'symmetric', 
                                        'periodic', 'smooth', 'periodization', 'reflect', 
                                        'antisymmetric', 'antireflect'. Defaults to 'periodization'.

    ## Returns:
        * int:  Wavelet energy of image.
    """
    # Validate wavelet & mode arguments
    assert wavelet in wavelist(),       f"Invalid argument for wavelet: {wavelet}. Valid options: {wavelist()}"
    assert mode in pywt.Modes.modes,    f"Inavlid argument for mode: {mode}. Valid options: {pywt.Modes.modes}"

    # Perform wavelet transform & calculate energy

    # Calculate energy
    return sum(sum(square(coeff)) for coeff in wavedec2(image, wavelet=wavelet, level=level, mode=mode))

def wavelet_entropy(
        image:      array, 
        wavelet:    str = "db2", 
        level:      int = 1,
        mode:       str = "periodization"
    ) -> float:
    """# Calculate the wavelet entropy of an image.

    ## Args:
        * image     (array):            Image for which the wavelet energy will be calculated.
        * wavelet   (str, optional):    Wavelet used in transform. Defaults to 'db2'.
        * level     (int, optional):    Level(s) of decomposition to perform. Defaults to 1.
        * mode      (str, optional):    Mode of padding. Options: 'zero', 'constant', 'symmetric', 
                                        'periodic', 'smooth', 'periodization', 'reflect', 
                                        'antisymmetric', 'antireflect'. Defaults to 'periodization'.

    ## Returns:
        * float:    Wavelet entropy of image.
    """
    # Initialize arrays to store all coefficients
    coefficients = []

    # For each level of coefficients computed...
    for coefficient in pywt.wavedec2(data = image, wavelet = wavelet, level = level, mode = mode):

        # Add their absolute values to the list
        coefficients.extend(abs(coefficient).ravel())

    # Convert them to an array
    coefficients = array(coefficients)
    
    # Normalize coefficients to probabilities
    probabilities = coefficients / sum(coefficients)
    
    # Return computed Shannon entropy
    return -sum(probabilities * log2(probabilities + finfo(float).eps))