"""Mexican Hat Wavelet transform and utilities."""

from logging    import Logger
from numpy      import array

from utils      import LOGGER

class MexicanHatWavelet():
    """Mexican Hat wavelet class."""
    
    _logger:            Logger =    LOGGER.getChild("mexican-hat-wavelet")
    
    def __init__(self, center: float = 2.0, edge: float = -0.4, corner: float = -0.1, amplitude: int = 2):
        """Initialize Mexican Hat wavelet kernel.

        Args:
            center      (float, optional):  Kernel's center value. Defaults to 2.0.
            edge        (float, optional):  Kernel's edge value. Defaults to -0.4.
            corner      (float, optional):  Kernel's corner value. Defaults to -0.1.
            amplitude   (int, optional):    Amplitude of wavelet (multiplier of kernel values). Defaults to 2.
        """
        self._logger.info(f"Initializing with (CENTER: {center}, EDGE: {edge}, CORNER: {corner}, AMPLITUDE: {amplitude})")
        
        self._kernel:   array =     array([
            [corner,    edge,   corner  ],
            [edge,      center, edge    ],
            [corner,    edge,   corner  ]
        ]) * amplitude
        
        self._logger.debug(f"Initialized kernel: \n{self._kernel}")