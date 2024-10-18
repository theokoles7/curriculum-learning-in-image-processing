"""Complexity metrics pertaining to edge density."""

from cv2    import Canny, COLOR_BGR2GRAY, cvtColor
from numpy  import asarray, ndarray, sum

def edge_density(
    image:  ndarray
) -> float:
    """Compute the Canny edge density of an image.

    Args:
        image (ndarray): Image sample.

    Returns:
        float: Edge density ratio of image sample.
    """
    # Convert to grayscale
    if image.shape[0] > 1:
        image:  ndarray =   cvtColor(asarray(image), COLOR_BGR2GRAY)
    
    # Count edges
    edges:      ndarray =   Canny(asarray(image), 100, 200)
    
    # Compute density
    return sum(edges == 255) / sum(edges == 0)
    