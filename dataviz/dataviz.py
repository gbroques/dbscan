from math import cos
from math import pi
from math import sin
from random import Random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_data(data: np.ndarray) -> None:
    """Plot data.

    Args:
        data: Data to plot.

    Returns:
        None
    """
    columns = ['x', 'y']
    df = pd.DataFrame(data, columns=columns)
    g = sns.lmplot(*columns,
                   data=df,
                   fit_reg=False,
                   legend=False)
    plt.show()


def plot_clusters(clusters: np.ndarray, labels: np.ndarray, components: np.ndarray, seed=0) -> None:
    """Plot cluster data as circles with a different color for each cluster.

    Plots core points as red crosses,
    and noise points as black points.

    Args:
        clusters: Cluster data to plot.
        labels: Labels of each point.
        components: Core points.
        seed: Seed for random number generator.
              Used to sample colors.

    Returns:
        None
    """
    noise = None
    contains_noise = any(labels == -1)
    if contains_noise:
        clusters, labels, noise = separate_noise(clusters, labels)
    num_clusters = len(np.unique(labels))
    contains_components = components.size > 0
    columns = ['x', 'y']
    data = get_data(clusters, labels, components, columns, noise)
    markers = get_markers(num_clusters, contains_noise, contains_components)
    palette = get_palette(num_clusters, contains_noise, contains_components, seed)
    g = sns.lmplot(*columns,
                   data=data,
                   markers=markers,
                   palette=palette,
                   fit_reg=False,
                   legend=False,
                   hue='labels',
                   scatter_kws={'linewidth': 1, 'edgecolor': 'w'})
    plt.show()


def separate_noise(clusters: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate noise points from clusters.

    Args:
        clusters: Cluster data.
        labels: The cluster each point belongs to.
                -1 is the label used for noise points.

    Returns:
        Clusters without noise, labels without noise, and noise points.
    """
    noise_indices = np.where(labels == -1)
    noise = clusters.take(noise_indices, axis=0)[0]
    clusters_without_noise = np.delete(clusters, noise_indices, axis=0)
    labels_without_noise = np.delete(labels, noise_indices)
    return clusters_without_noise, labels_without_noise, noise


def get_data(clusters, labels, components, columns, noise=None) -> pd.DataFrame:
    """Construct a DataFrame object to plot.

    Args:
        clusters: The cluster data.
        labels: Which cluster each point belongs to.
        components: Core points.
        columns: Labels for each column of data.
        noise: Noise points.

    Returns:
        Data frame for plotting.
    """
    df = pd.DataFrame(clusters, columns=columns)
    df['labels'] = pd.Series(labels, index=df.index)  # Add labels as a column for coloring
    df = append_data_frame(df, columns, components, 'core_point')
    if noise is not None:
        df = append_data_frame(df, columns, noise, 'noise_point')
    return df


def append_data_frame(df, columns, data, label) -> pd.DataFrame:
    if data.size == 0:
        return df
    components_df = pd.DataFrame(data, columns=columns)
    components_df['labels'] = [label for _ in range(len(data))]
    return df.append(components_df, ignore_index=True)


def get_markers(num_clusters: int, contains_noise: bool, contains_components: bool) -> List[str]:
    """Generate the marks for the plot.

    Uses circles 'o' for points,
    crosses 'x' for core points,
    and points ',' for noise points.

    Args:
        num_clusters: The number of clusters.
        contains_noise: Whether the dataset contains noise points.
        contains_components: Whether the dataset contains core points.

    Returns:
        A list of markers.
    """
    markers = ['o' for _ in range(num_clusters)]
    if contains_components:
        markers.append('x')  # Use crosses 'x' for core points
    if contains_noise:
        markers.append('.')  # Use points '.' for noise points
    return markers


def get_palette(num_clusters: int, contains_noise: bool, contains_components: bool, seed=0) -> List[str]:
    """Generates a color palette for the plot.

    Uses random colors for different clusters,
    and reserves red for centroids.

    Args:
        num_clusters: The number of clusters.
        contains_noise: Whether the dataset contains noise.
        contains_components: Whether the dataset contains core points.
        seed: Seed for random number generator.

    Returns:
        A list of colors used for plotting.
    """
    random = Random(seed)
    all_colors = ['b', 'g', 'c', 'm', 'orange',
                  'darkturquoise', 'mediumpurple', 'tomato']
    palette = random.sample(all_colors, num_clusters)
    if contains_components:
        palette.append('red')  # Reserve red color for core points
    if contains_noise:
        palette.append('k')  # Reserve black color for noise points
    return palette


def generate_clusters(num_clusters: int,
                      num_points: int,
                      spread: float,
                      x_bounds: Tuple[float, float],
                      y_bounds: Tuple[float, float],
                      seed=None) -> List[List]:
    """Generate random data for clustering.

    Source:
    https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x

    Args:
        num_clusters: The number of clusters to generate.
        num_points: The number of points to generate.
        spread: The spread of each cluster. Decrease for tighter clusters.
        x_bounds: The bounds for possible values of X.
        y_bounds: The bounds for possible values of Y.
        seed: Seed for the random number generator.

    Returns:
        K clusters consisting of N points.
    """
    random = Random(seed)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
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
