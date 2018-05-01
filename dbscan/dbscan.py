from typing import List

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist

from .distance_matrix import condensed_to_square
from .distance_matrix import get_condensed_indices

DEFAULT_LABEL = -2
NOISE_LABEL = -1


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
        self._check_eps(eps)
        self._check_min_samples(min_samples)
        self._eps = eps
        self._min_samples = min_samples
        self._distance_matrix = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None

    @staticmethod
    def _check_eps(eps: float) -> None:
        if eps <= 0:
            raise ValueError('eps must be a positive floating point number. {} given.'.format(eps))

    @staticmethod
    def _check_min_samples(min_samples: int) -> None:
        if min_samples <= 0 or not isinstance(min_samples, int):
            raise ValueError('min_samples must be a positive integer. {} given.'.format(min_samples))

    def fit(self, data: ndarray) -> 'DBSCAN':
        """Perform DBSCAN clustering on features.

        Args:
            data: List of features.

        Returns:
            self
        """
        self._distance_matrix = pdist(data)  # Condensed pair-wise distance matrix
        num_samples = len(data)
        self.core_sample_indices_ = self._get_core_sample_indices(num_samples)
        self.components_ = self._get_components(data)
        self.labels_ = self._get_labels(data)
        return self

    def _get_components(self, data: ndarray) -> ndarray:
        if self.core_sample_indices_.size == 0:
            return np.array([])
        else:
            return data.take(self.core_sample_indices_, axis=0)

    def _get_labels(self, data: ndarray) -> ndarray:
        labels = [DEFAULT_LABEL for _ in data]
        num_samples = len(data)
        current_cluster_label = -1
        for i in range(0, num_samples):
            if labels[i] != DEFAULT_LABEL:
                continue  # Skip points with labels

            neighborhood = self._get_neighboring_point_indices(i, num_samples)

            num_points_in_neighborhood = get_num_points_in_neighborhood(neighborhood)
            if num_points_in_neighborhood < self._min_samples:
                labels[i] = NOISE_LABEL
            else:
                current_cluster_label += 1
                self._grow_cluster(labels, i, neighborhood, current_cluster_label)

        return np.array(labels)

    def _grow_cluster(self,
                      labels: List[int],
                      seed_point_index: int,
                      neighborhood: ndarray,
                      cluster_label: int) -> None:
        """Keep growing the cluster until there's no more core points.

        Args:
            labels: Cluster labels.
            seed_point_index: Index of point from which to grow the cluster.
            neighborhood: Indices of points within the neighborhood of the seed point.
            cluster_label: The current cluster label of the seed point.

        Returns:
            None
        """
        labels[seed_point_index] = cluster_label
        i = 0
        while i < len(neighborhood):
            neighboring_point = neighborhood[i]

            if labels[neighboring_point] == NOISE_LABEL:
                labels[neighboring_point] = cluster_label  # Label border point
            elif labels[neighboring_point] == DEFAULT_LABEL:
                labels[neighboring_point] = cluster_label

                num_samples = len(labels)
                neighboring_neighborhood = self._get_neighboring_point_indices(neighboring_point, num_samples)

                num_points_in_neighborhood = get_num_points_in_neighborhood(neighborhood)

                # Grow the neighborhood of the cluster
                if num_points_in_neighborhood >= self._min_samples:
                    neighborhood = np.append(neighborhood, neighboring_neighborhood)

            i += 1

    def _get_neighboring_point_indices(self, neighboring_point: int, num_samples: int):
        """Convenience method to get neighboring point indices."""
        args = [neighboring_point, num_samples, self._distance_matrix, self._eps]
        return get_neighboring_point_indices(*args)

    def _get_core_sample_indices(self, num_samples: int) -> ndarray:
        """Get the indices of each core point.

        Args:
            num_samples: Number of points in the dataset.

        Returns:
            Indices of each core point.
        """
        core_sample_indices = []
        for i in range(num_samples):
            distances = get_distances_from_other_points(i, num_samples, self._distance_matrix)
            less_than_eps = np.where(distances < self._eps, 1, 0)
            num_less_than_eps = np.count_nonzero(less_than_eps) + 1  # + 1 includes the point itself
            if num_less_than_eps >= self._min_samples:
                core_sample_indices.append(i)
        return np.array(core_sample_indices)


def get_distances_from_other_points(point_index: int, num_samples: int, distance_matrix: ndarray):
    """Get distances from point i to other points in the dataset.

    Args:
        point_index: The index of the point.
        num_samples: The number of points.
        distance_matrix: Condensed distance matrix.

    Returns:
        Distance from point i to other points in the dataset.
    """
    condensed_indices = get_condensed_indices(point_index, num_samples)
    return distance_matrix.take(condensed_indices)


def get_neighboring_point_indices(point_index: int, num_samples: int, distance_matrix: ndarray, eps: float) -> ndarray:
    """Get the indices of points within the neighborhood of point i as defined by eps.

    Args:
        point_index: Index of the point.
        num_samples: The total number of samples.
        distance_matrix: Condensed pair-wise distance matrix.
        eps: The maximum distance between two samples for them to be considered as in the same neighborhood.

    Notes
        Does not include the point itself.

    Returns:
        A list of neighboring point indices as defined by eps.
    """
    condensed_indices = get_condensed_indices(point_index, num_samples)
    condensed_indices_np = np.array(condensed_indices)
    indices = np.where(distance_matrix.take(condensed_indices) < eps)
    condensed_indices_within_eps = condensed_indices_np[indices]
    square_indices_within_eps = map(lambda e: condensed_to_square(e, num_samples), condensed_indices_within_eps)
    indices = [[e for e in t if e != point_index][0] for t in square_indices_within_eps]
    return np.array(indices)


def get_num_points_in_neighborhood(neighborhood: ndarray):
    """Get the number of points in the neighborhood of a given point.

    Args:
        neighborhood: The neighborhood from get_neighboring_point_indices.

    Notes
        Add 1 because get_neighboring_point_indices does not include the point itself.

    Returns:
        The length of the neighborhood plus 1.
    """
    return len(neighborhood) + 1
