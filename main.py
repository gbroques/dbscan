import numpy as np

from dataviz import generate_clusters
from dataviz import plot_clusters
from dbscan import DBSCAN


def main():
    num_clusters = 4
    clusters = generate_data(num_clusters, seed=1)
    dbscan = DBSCAN(eps=7, min_samples=5)
    dbscan.fit(clusters)
    plot_clusters(clusters, dbscan.labels_, dbscan.components_)


def generate_data(num_clusters: int, seed=None) -> np.ndarray:
    num_points = 20
    spread = 7
    bounds = (1, 100)
    clusters = generate_clusters(num_clusters, num_points, spread, bounds, bounds, seed)
    return np.array(clusters)


if __name__ == '__main__':
    main()
