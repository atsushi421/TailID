"""Data processing utilities for TailID algorithm.

This module provides helper functions for data processing operations
used in the TailID algorithm for detecting ID-sensitive points in the
tail of probability distributions.
"""

import numpy as np
from numpy.typing import NDArray


def quantile(data: NDArray[np.floating], percentile: float) -> float:
    """Compute the value at specified percentile in data.

    Args:
        data: Sample data array.
        percentile: Percentile value between 0 and 1.

    Returns:
        The value at the specified percentile.
    """
    return float(np.quantile(data, percentile))


def select_candidates(
    data: NDArray[np.floating], percentile: float
) -> NDArray[np.floating]:
    """Extract ordered set of values at or above the specified percentile.

    Args:
        data: Sample data array.
        percentile: Percentile value between 0 and 1.

    Returns:
        Ordered array of candidate values (sorted in ascending order).
    """
    threshold = quantile(data, percentile)
    candidates = data[data >= threshold]
    return np.sort(candidates)


def excess_set(
    data: NDArray[np.floating], threshold: float
) -> NDArray[np.floating]:
    """Compute threshold exceedances (values above threshold minus threshold).

    Args:
        data: Sample data array.
        threshold: Threshold value for computing excesses.

    Returns:
        Array of exceedances (x_i - threshold for all x_i > threshold).
    """
    exceedances = data[data > threshold] - threshold
    return exceedances
