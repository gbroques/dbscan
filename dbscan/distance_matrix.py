"""
Functions for converting between square matrix positions (i, j),
and condensed distance matrix indices.
"""

import math
from itertools import chain
from itertools import repeat
from typing import List


def square_to_condensed(i: int, j: int, n: int):
    """Convert a square matrix position (i, j) to a condensed distance matrix index.

    Args:
        i: Index i.
        j: Index j.
        n: The dimension of the matrix.

    See Also:
        https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

    Returns:
        Condensed index.
    """
    assert i != j, 'No diagonal elements in condensed matrix'
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) / 2 + i - 1 - j


def condensed_to_square(k, n):
    """Convert a condensed distance matrix index to a square matrix position (i, j).

    Args:
        k: Condensed index.
        n: The dimension of the matrix.

    See Also:
        https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

    Returns:
        Condensed index.
    """
    i = row_index(k, n)
    j = column_index(k, i, n)
    return i, j


def get_condensed_indices(i: int, n: int) -> List[int]:
    square_indices = zip(repeat(i, n), chain(range(0, i), range(i + 1, n)))
    condensed_indices = [square_to_condensed(*square_index, n) for square_index in square_indices]
    return condensed_indices


def row_index(k, n):
    """
    See Also:
        https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

    Args:
        k: Condensed index.
        n: Dimension of distance matrix.

    Returns:
        Row index.
    """
    return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n ** 2 - 4 * n - 7) ** 0.5 + 2 * n - 1) - 1))


def element_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) / 2


def column_index(k, i, n):
    return int(n - element_in_i_rows(i + 1, n) + k)
