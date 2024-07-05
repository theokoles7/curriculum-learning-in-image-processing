"""Basic CNN model."""

from torch                  import Tensor
from torch.nn               import Conv2d, Linear, MaxPool2d, Module
from torch.nn.functional    import relu

from utils                  import ARGS, LOGGER

class CNN(Module):
    """Basic CNN model."""

    # Initialize logger
    _logger =       LOGGER.getChild('cnn')

    def __init__(self, channels_in: int, channels_out: int, dim: int):
        """Initialize Normal CNN model.

        Args:
            channels_in     (int):  Input channels
            channels_out    (int):  Output channels
            dim             (int):  Dimension of image (relevant for reshaping, post-convolution)
        """
        super(CNN, self).__init__()

        # Convolving layers
        self.conv1:         Conv2d =    Conv2d(in_channels = channels_in, out_channels =  32, kernel_size=3, padding=1)
        self.conv2:         Conv2d =    Conv2d(in_channels =          32, out_channels =  64, kernel_size=3, padding=1)
        self.conv3:         Conv2d =    Conv2d(in_channels =          64, out_channels = 128, kernel_size=3, padding=1)
        self.conv4:         Conv2d =    Conv2d(in_channels =         128, out_channels = 256, kernel_size=3, padding=1)

        # Max pooling layers
        self.pool1:         MaxPool2d = MaxPool2d(kernel_size=2, stride=2)
        self.pool2:         MaxPool2d = MaxPool2d(kernel_size=2, stride=2)
        self.pool3:         MaxPool2d = MaxPool2d(kernel_size=2, stride=2)
        self.pool4:         MaxPool2d = MaxPool2d(kernel_size=2, stride=2)

        # Fully-connected layer
        self.fc:            Linear =    Linear(in_features = dim**2, out_features = 1024)

        # Classifier
        self.classifier:    Linear =    Linear(in_features = 1024, out_features = channels_out)

    def forward(self, X: Tensor) -> Tensor:
        """Feed input through network and produce output.

        Args:
            X   (Tensor):   Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # INPUT LAYER =============================================================================
        self._logger.debug(f"Input shape:   {X.shape}")

        # LAYER 1 =================================================================================
        x1 =    self.conv1(X)
        self._logger.debug(f"Layer 1 shape: {x1.shape}")

        # LAYER 2 =================================================================================
        x2 =    self.conv2(relu(self.pool1(x1)))
        self._logger.debug(f"Layer 2 shape: {x2.shape}")

        # LAYER 3 =================================================================================
        x3 =    self.conv3(relu(self.pool2(x2)))
        self._logger.debug(f"Layer 3 shape: {x3.shape}")

        # LAYER 4 =================================================================================
        x4 =    self.conv4(relu(self.pool3(x3)))
        self._logger.debug(f"Layer 4 shape: {x4.shape}")

        # OUTPUT LAYER ============================================================================
        output =    relu(self.pool4(x4))
        self._logger.debug(f"Output shape:  {output.shape}")

        # Return classified output
        return self.classifier(relu(self.fc(output.view(output.size(0), -1))))