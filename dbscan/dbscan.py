import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist

from .distance_matrix import condensed_to_square
from .distance_matrix import get_condensed_indices


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
        distance_matrix = pdist(data)  # Condensed pair-wise distance matrix
        n = len(data)
        self.core_sample_indices_ = self._get_core_sample_indices(n, distance_matrix)
        self.components_ = self._get_components(data)
        self.labels_ = self._get_labels(data, distance_matrix)
        return self

    def _get_components(self, data: ndarray) -> ndarray:
        if self.core_sample_indices_.size == 0:
            return np.array([])
        else:
            return data.take(self.core_sample_indices_, axis=0)

    def _get_labels(self, data: ndarray, distance_matrix: ndarray) -> ndarray:
        labels = [-1 for _ in data]
        n = len(data)
        current_cluster_label = -1
        for i, core_point in zip(self.core_sample_indices_, self.components_):
            if labels[i] == -1:
                current_cluster_label += 1
                labels[i] = current_cluster_label
            neighborhood = get_indices_of_points_within_eps(i, n, distance_matrix, self._eps)
            for point_i in neighborhood:
                if labels[point_i] == -1:
                    labels[point_i] = current_cluster_label
        return np.array(labels)

    def _get_core_sample_indices(self, n: int, distance_matrix: ndarray) -> ndarray:
        """Get the indices of each core point.

        Args:
            n: Number of points in the dataset.
            distance_matrix: Condensed distance matrix.

        Returns:
            Indices of each core point.
        """
        core_sample_indices = []
        for i in range(n):
            distances = get_distances_from_other_points(i, n, distance_matrix)
            less_than_eps = np.where(distances < self._eps, 1, 0)
            num_less_than_eps = np.count_nonzero(less_than_eps) + 1  # + 1 includes the point itself
            if num_less_than_eps >= self._min_samples:
                core_sample_indices.append(i)
        return np.array(core_sample_indices)


def get_distances_from_other_points(i: int, n: int, distance_matrix: ndarray):
    """Get distances from point i to other points in the dataset.

    Args:
        i: The index of the point.
        n: The number of points.
        distance_matrix: Condensed distance matrix.

    Returns:
        Distance from point i to other points in the dataset.
    """
    condensed_indices = get_condensed_indices(i, n)
    return distance_matrix.take(condensed_indices)


def get_indices_of_points_within_eps(i: int, n: int, distance_matrix: ndarray, eps: float) -> ndarray:
    condensed_indices = get_condensed_indices(i, n)
    condensed_indices_np = np.array(condensed_indices)
    indices = np.where(distance_matrix.take(condensed_indices) < eps)
    condensed_indices_within_eps = condensed_indices_np[indices]
    square_indices_within_eps = map(lambda e: condensed_to_square(e, n), condensed_indices_within_eps)
    indices = [[e for e in t if e != i][0] for t in square_indices_within_eps]
    return np.array(indices)
