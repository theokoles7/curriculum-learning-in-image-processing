"""Initialize dataset based on selection."""

from datasets.cifar10 import Cifar10
from datasets.cifar100 import Cifar100


def get_dataset(
    dataset: str,
    path: str = "data",
    batch_size: int = 16,
    curriculum: str = None,
    by_batch: bool = False,
    sort_mean: bool = False,
) -> Cifar10 | Cifar100:
    """Initialize and return dataset loaders based on selection.

    Args:
        dataset     (str):              Dataset selection.
        path        (str, optional):    Path at which dataset can be located/downlaoded. Defaults to "data/{dataset_name}".
        batch_size  (int, optional):    Dataset loader batch size. Defaults to 16.
        curriculum  (str, optional):    Curriculum selection. Defaults to None.
        by_batch    (bool, optional):   Sort individual batches, instead of entire dataset.

    Returns:
        Cifar10 | Cifar100: CIFAR10 or Cifar100 dataset.
    """
    # Match datsset
    match dataset:

        # Return valid dataset choices
        case "cifar10":
            return Cifar10(
                path=f"{path}/cifar10",
                batch_size=batch_size,
                curriculum=curriculum,
                by_batch=by_batch,
                sort_mean=sort_mean,
            )
        case "cifar100":
            return Cifar100(
                path=f"{path}/cifar100",
                batch_size=batch_size,
                curriculum=curriculum,
                by_batch=by_batch,
                sort_mean=sort_mean,
            )

        # Raise warning for others
        case _:
            raise ValueError(f"{dataset} is not a valid dataset seleciton.")
