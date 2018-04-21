from math import cos
from math import pi
from math import sin
from random import Random
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_clusters(clusters: List[List], labels: List[int], centroids: List[List], seed=0) -> None:
    """Plot cluster data.

    Args:
        clusters: Cluster data to plot.
        labels: Labels of each point.
        centroids: Center point of each cluster.
        seed: Seed for random number generator.
              Used to sample colors.

    Returns:
        None
    """
    columns = ['x', 'y']
    num_clusters = len(set(labels))
    data = get_data(clusters, labels, centroids, columns)
    markers = get_markers(num_clusters)
    palette = get_palette(num_clusters, seed)
    g = sns.lmplot(*columns,
                   data=data,
                   markers=markers,
                   palette=palette,
                   fit_reg=False,
                   legend=False,
                   hue='labels',
                   scatter_kws={'linewidth': 1, 'edgecolor': 'w'})
    plt.show()


def get_data(clusters, labels, centroids, columns) -> pd.DataFrame:
    """Construct a DataFrame object to plot.

    Args:
        clusters: The cluster data.
        labels: Which cluster each point belongs to.
        centroids: The center point of each cluster.
        columns: Labels for each column of data.

    Returns:

    """
    df = pd.DataFrame(clusters, columns=columns)
    df['labels'] = pd.Series(labels, index=df.index)  # Add labels as a column for coloring
    centroids_df = pd.DataFrame(centroids, columns=columns)
    centroids_df['labels'] = ['centroid' for _ in range(len(centroids))]
    df = df.append(centroids_df, ignore_index=True)
    return df


def get_markers(num_clusters) -> List[str]:
    """Generate the marks for the plot.

    Uses circles 'o' for points,
    and crosses 'x' for centroids.

    Args:
        num_clusters: The number of clusters.

    Returns:
        A list of markers.
    """
    markers = ['o' for _ in range(num_clusters)]
    markers.append('x')  # Reserve 'x' for centroids
    return markers


def get_palette(num_clusters, seed=0) -> List[str]:
    """Generates a color palette for the plot.

    Uses random colors for different clusters,
    and reserves red for centroids.

    Args:
        num_clusters: The number of clusters.
        seed: Seed for random number generator.

    Returns:

    """
    random = Random(seed)
    all_colors = ['b', 'g', 'c', 'm', 'orange']
    palette = random.sample(all_colors, num_clusters)
    palette.append('red')  # Reserve red color for centroids
    return palette


def generate_clusters(num_clusters: int,
                      num_points: int,
                      spread: float,
                      bound_for_x: Tuple[float, float],
                      bound_for_y: Tuple[float, float],
                      seed=None) -> List[List]:
    """Generate random data for clustering.

    Source:
    https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x

    Args:
        num_clusters: The number of clusters to generate.
        num_points: The number of points to generate.
        spread: The spread of each cluster. Decrease for tighter clusters.
        bound_for_x: The bounds for possible values of X.
        bound_for_y: The bounds for possible values of Y.
        seed: Seed for the random number generator.

    Returns:
        K clusters consisting of N points.
    """
    random = Random(seed)
    x_min, x_max = bound_for_x
    y_min, y_max = bound_for_y
    num_points_per_cluster = int(num_points / num_clusters)
    clusters = []
    for _ in range(num_clusters):
        x = x_min + (x_max - x_min) * random.random()
        y = y_min + (y_max - y_min) * random.random()
        clusters.extend(generate_cluster(num_points_per_cluster, (x, y), spread, seed))
    return clusters


def generate_cluster(num_points: int, center: Tuple[float, float], spread: float, seed=None) -> List[List]:
    """Generates a cluster of random points.

    Source:
    https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x

    Args:
        num_points: The number of points for the cluster.
        center: The center of the cluster.
        spread: How tightly to cluster the data.
        seed: Seed for the random number generator.

    Returns:
        A random cluster of consisting of N points.
    """
    x, y = center
    seed = (seed + y) * x  # Generate different looking clusters if called from generate_clusters
    random = Random(seed)
    points = []
    for i in range(num_points):
        theta = 2 * pi * random.random()
        s = spread * random.random()
        point = [x + s * cos(theta), y + s * sin(theta)]
        points.append(point)
    return points
