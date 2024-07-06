"""CIFAR-100 dataset & utilities."""

from logging                import Logger
from numpy                  import argsort, ndarray
from torch.utils.data       import DataLoader
from torchvision.datasets   import CIFAR100
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from curriculums            import curriculums, CurriculumSampler
from utils                  import LOGGER

class Cifar100():
    """This dataset is just like the CIFAR-10 
    (https://www.cs.toronto.edu/~kriz/cifar.html), except it has 100 
    classes containing 600 images each. There are 500 training images 
    and 100 testing images per class. The 100 classes in the CIFAR-100 
    are grouped into 20 superclasses. Each image comes with a "fine" 
    label (the class to which it belongs) and a "coarse" label (the 
    superclass to which it belongs). 
    """

    # Initialize logger
    _logger:    Logger =    LOGGER.getChild('cifar10-dataset')

    def __init__(self, 
            path:       str =   "data", 
            batch_size: int =   16, 
            curriculum: str =   None,
            by_batch:   bool =  False
        ):
        """Initialize Cifar100 dataset loaders.

        Args:
            * path          (str, optional):    Path at which dataset can be located/downloaded. Defaults to 'data'.
            * batch_size    (int, optional):    Dataset batch size. Defaults to 16.
            * curriculum    (str, optional):    Curriculum by which dataset will be arranged. Defaults to None.
        """
        # Create transform
        self._logger.info("Initializing transform")
        transform = Compose([
            Resize(32),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

        # Verify train data
        self._logger.info("Verifying/downloading train data")
        train_data = CIFAR100(
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
        test_data = CIFAR100(
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
        self.num_classes = 100
        self.channels_in =   3
        self.dim =          32

        self._logger.debug(f"DATASET: {self} | CLASSES: {self.num_classes} | CHANNELS: {self.channels_in} | DIM: {self.dim}")
        self._logger.debug(f"TRAIN LOADER:\n{vars(self.train_loader)}")
        self._logger.debug(f"TEST LOADER:\n{vars(self.test_loader)}")

    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Fetch data loaders.

        Returns:
            typing.Tuple[data.DataLoader, data.DataLoader]: Cifar100 train & test loaders
        """
        return self.train_loader, self.test_loader
    
    def __str__(self) -> str:
        """Provide string format of Cifar100 dataset object.

        Returns:
            str: String format of Cifar100 dataset
        """
        return f"Cifar100 dataset ({self.num_classes} classes)"