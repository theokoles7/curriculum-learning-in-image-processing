"""Initialize dataset based on selection."""

from datasets.cifar10   import Cifar10

def get_dataset(
    dataset:    str,
    path:       str =   "data",
    batch_size: int =   16,
    curriculum: str =   None
) -> Cifar10:
    """Initialize and return dataset loaders based on selection.

    Args:
        dataset (str): Dataset selection.
        path (str, optional): Path at which dataset can be located/downlaoded. Defaults to "data/{dataset_name}".
        batch_size (int, optional): Dataset loader batch size. Defaults to 16.
        curriculum (str, optional): Curriculum selection. Defaults to None.

    Returns:
        Cifar10: CIFAR10 dataset.
    """
    # Match datsset
    match dataset:

        case "cifar10": return Cifar10(path =   f"{path}/cifar10", batch_size = batch_size, curriculum = curriculum)

        case _: raise ValueError(f"{dataset} is not a valid dataset seleciton.")