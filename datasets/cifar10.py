"""CIFAR-10 dataset & utilities."""

from logging                import Logger
from numpy                  import argsort, ndarray
from torch.utils.data       import DataLoader
from torchvision.datasets   import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from curriculums            import curriculums, CurriculumSampler
from utils                  import LOGGER

class Cifar10():
    """The CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 
    colour images in 10 classes, with 6000 images per class. There are 50000 training images and 
    10000 test images.
    """

    # Initialize logger
    _logger:    Logger =    LOGGER.getChild('cifar10-dataset')

    def __init__(self, 
            path:       str =   "data", 
            batch_size: int =   16, 
            curriculum: str =   None,
            by_batch:   bool =  False
        ):
        """Initialize Cifar10 dataset loaders.

        Args:
            * path          (str, optional):    Path at which dataset can be located/downloaded. Defaults to 'data'.
            * batch_size    (int, optional):    Dataset batch size. Defaults to 16.
            * curriculum    (str, optional):    Curriculum by which dataset will be arranged. Defaults to None.
        """
        # Record curriculum for collation function
        self._curriculum: str =  curriculum
        
        # Create transform
        self._logger.info("Initializing transform")
        transform = Compose([
            Resize(32),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify train data
        self._logger.info("Verifying/downloading train data")
        train_data = CIFAR10(
            root =          path,
            download =      True,
            train =         True,
            transform =     transform
        )

        # If curriculum was specified
        if curriculum and not by_batch:
            self._logger.info(f"Sorting training data by {curriculum}")

            # Determine sorted indeces
            sorted_indeces: ndarray =   argsort([curriculums[curriculum](image) for image in train_data.data])

            # Re-arrange images and labels
            train_data.data =       train_data.data[sorted_indeces]
            train_data.targets =    [train_data.targets[i] for i in sorted_indeces]

        # Verify test data
        self._logger.info("Verifying/downloading test data")
        test_data = CIFAR10(
            root =          path,
            download =      True,
            train =         False,
            transform =     transform
        )
        
        # If curriculum was specified by batch
        if curriculum and by_batch:
            self._logger.info("Creating train data loader with batch sampler.")
            
            # Create dataloader with batch sampler
            self.train_loader = DataLoader(
                train_data,
                pin_memory =    True,
                num_workers =   4,
                batch_sampler = CurriculumSampler(train_data, batch_size = batch_size, curriculum = curriculums[curriculum]) if curriculum and by_batch else None
            )
            
        # Otherwise, create ordinary data loader
        else:
            self._logger.info("Creating train data loader.")
            
            # Create dataloader with batch sampler
            self.train_loader = DataLoader(
                train_data,
                batch_size =    batch_size,
                pin_memory =    True,
                num_workers =   4,
                shuffle =       False if curriculum else True,
                drop_last =     False,
            )

        # Create testing loader
        self._logger.info("Creating test data loader.")
        self.test_loader = DataLoader(
            test_data,
            batch_size =    batch_size,
            pin_memory =    True,
            num_workers =   4,
            shuffle =       True,
            drop_last =     False
        )

        # Define parameters
        self.num_classes =  10
        self.channels_in =   3
        self.dim =          32

        self._logger.debug(f"DATASET: {self} | CLASSES: {self.num_classes} | CHANNELS: {self.channels_in} | DIM: {self.dim}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self.train_loader)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self.test_loader)}")

    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Cifar10 train & test loaders
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of Cifar10 dataset object.

        Returns:
            str: String format of Cifar10 dataset
        """
        return f"Cifar10 dataset ({self.num_classes} classes)"