"""Curriculum batch sampler."""

from random import shuffle
from typing import Callable, Iterator

from numpy import argsort, array_split, ndarray, abs, mean
from torch.utils.data import Dataset, Sampler


class CurriculumSampler(Sampler):

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        curriculum: Callable,
        sort_mean: bool = False,
    ):
        """# Initialize custom batch sampler object.

        ## Args:
            * data_source   (Dataset):  Dataset source.
            * batch_size    (int):      Batch size.
            * curriculum    (Callable): Curriculum function.
        """
        # Initialize attributes
        self._data_source: Dataset = data_source
        self._batch_size: int = batch_size
        self._curriculum: Callable = curriculum
        self._sort_mean: bool = sort_mean

    def __len__(self) -> int:
        """# Provide length of series of batches.

        ## Returns:
            * int:  Number of batches.
        """
        return len(self._data_source) // self._batch_size

    def __iter__(self) -> Iterator:

        # Split data into partitions
        data_partitions: list = array_split(
            self._data_source.data, len(self._data_source) // self._batch_size
        )

        # Sort each partition
        for partition in data_partitions:
            # Calculate values
            curriculum_values = [self._curriculum(image) for image in partition]

            # Sort from middle if needed
            if self._sort_mean:
                curriculum_values = abs(curriculum_values - mean(curriculum_values))

            yield (argsort(curriculum_values))
