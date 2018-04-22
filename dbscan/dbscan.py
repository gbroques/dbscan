from itertools import repeat
from itertools import chain
from numpy import ndarray
from scipy.spatial.distance import pdist


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

    Args:
        eps: The maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples: The number of samples in a neighborhood for a point to be considered a core point.
                           This includes the point itself.

    Attributes:
        core_sample_indices_ (list): Indices of core samples.
                                     Shape [n_core_samples]
        components_ (list): Copy of each core sample found by training.
                            Shape [n_core_samples, n_features]
        labels_ (list): Cluster labels for each point in the dataset given to fit().
                        Noisy samples are given the label -1.
                        Shape [n_samples]
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self._eps = eps
        self._min_samples = min_samples
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None

    def fit(self, data: ndarray) -> 'DBSCAN':
        """Perform DBSCAN clustering on features.

        Args:
            data: List of features.

        Returns:
            self
        """
        distance_matrix = pdist(data)
        return self


def get_distances_from_other_points(i: int, n: int, distance_matrix: ndarray):
    """Get distances from point i to other points in the dataset.

    Args:
        i: The index of the point.
        n: The number of points.
        distance_matrix: Condensed distance matrix.

    Returns:
        Distance from point i to other points in the dataset.
    """
    square_indices = zip(repeat(i, n), chain(range(0, i), range(i + 1, n)))
    condensed_indices = [square_to_condensed(*square_index, n) for square_index in square_indices]
    return distance_matrix.take(condensed_indices)


def square_to_condensed(i: int, j: int, n: int):
    """Maps a square index to a condensed index for a given matrix position (i, j).

    Args:
        i: Index i.
        j: Index j.
        n: The dimension of the matrix.

    Returns:
        Condensed index.
    """
    assert i != j, 'No diagonal elements in condensed matrix'
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) / 2 + i - 1 - j
