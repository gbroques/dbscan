import unittest

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist

from dbscan import DBSCAN
from dbscan.dbscan import get_distances_from_other_points
from dbscan.dbscan import get_indices_of_points_within_eps


class DBSCANTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eps = 1.5
        cls.min_samples = 3

    @staticmethod
    def get_two_clusters() -> ndarray:
        """Two clusters in a 2-dimensional plane.
        ^
        | *        o
        |            o
        | x            o
        |   x
        |     x
        +-------------->
        x - Denotes cluster 1
        o - Denotes cluster 2
        * - Denotes noise point
        """
        return np.array([[1, 3], [2, 2], [3, 1],
                         [6, 8], [7, 7], [8, 6],
                         [1, 8]])

    def fit(self):
        data = self.get_two_clusters()
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan.fit(data)
        return dbscan

    def test_core_sample_indices(self):
        expected_core_sample_indices = np.array([1, 4])
        dbscan = self.fit()
        np.testing.assert_equal(expected_core_sample_indices, dbscan.core_sample_indices_)

    def test_components(self):
        expected_components = np.array([[2, 2], [7, 7]])
        dbscan = self.fit()
        np.testing.assert_equal(expected_components, dbscan.components_)

    def test_labels(self):
        expected_labels = np.array([0, 0, 0, 1, 1, 1, -1])
        dbscan = self.fit()
        np.testing.assert_equal(expected_labels, dbscan.labels_)

    def test_fit_with_small_eps(self):
        expected_core_sample_indices = np.array([])
        expected_components = np.array([])
        expected_labels = np.array([-1, -1, -1, -1, -1, -1, -1])

        data = self.get_two_clusters()
        dbscan = DBSCAN(eps=0.1, min_samples=3)
        dbscan.fit(data)

        np.testing.assert_equal(expected_core_sample_indices, dbscan.core_sample_indices_)
        np.testing.assert_equal(expected_components, dbscan.components_)
        np.testing.assert_equal(expected_labels, dbscan.labels_)

    def test_raises_error_for_invalid_eps(self):
        with self.assertRaises(ValueError):
            DBSCAN(eps=0)

    def test_raises_error_for_invalid_min_samples(self):
        with self.assertRaises(ValueError):
            DBSCAN(min_samples=0)

    def test_get_distances_from_other_points_with_index_zero(self):
        expected_distances_from_other_points = np.array([1.4142135, 2.8284271, 7.0710678,
                                                         7.2111025, 7.6157731, 5.0000000])
        self.assert_distances_almost_equal(0, expected_distances_from_other_points)

    def test_get_distances_from_other_points_with_index_one(self):
        expected_distances_from_other_points = np.array([1.4142135, 1.4142135, 7.2111025,
                                                         7.0710678, 7.2111025, 6.0827625])
        self.assert_distances_almost_equal(1, expected_distances_from_other_points)

    def test_get_distances_from_other_points_with_index_two(self):
        expected_distances_from_other_points = np.array([2.8284271, 1.4142136, 7.6157731,
                                                         7.2111026, 7.0710678, 7.2801099])
        self.assert_distances_almost_equal(2, expected_distances_from_other_points)

    def test_get_distances_from_other_points_with_index_three(self):
        expected_distances_from_other_points = np.array([7.0710678, 7.2111025, 7.6157731,
                                                         1.4142135, 2.8284271, 5.0000000])
        self.assert_distances_almost_equal(3, expected_distances_from_other_points)

    def assert_distances_almost_equal(self, index: int, expected_distances_from_other_points):
        distances_from_other_points = self.distances_from_other_points(index)
        np.testing.assert_almost_equal(expected_distances_from_other_points, distances_from_other_points)

    def distances_from_other_points(self, point_index: int):
        data = self.get_two_clusters()
        distance_matrix = pdist(data)
        num_points = len(data)
        return get_distances_from_other_points(point_index, num_points, distance_matrix)

    def test_get_indices_of_points_within_eps(self):
        expected_indices = np.array([0, 2])
        data = self.get_two_clusters()
        n = len(data)
        distance_matrix = pdist(data)
        indices = get_indices_of_points_within_eps(1, n, distance_matrix, self.eps)
        assert np.array_equal(expected_indices, indices)


if __name__ == '__main__':
    unittest.main()
