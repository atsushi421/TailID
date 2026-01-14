"""GPD fitting and statistical functions for TailID algorithm.

This module provides functions for fitting the Generalized Pareto Distribution
(GPD) and computing confidence intervals for the Extreme Value Index (EVI),
which are core components of the TailID algorithm.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto, norm


def fit_gpd_evi(excess_data: NDArray[np.floating]) -> float:
    """Fit GPD and estimate the Extreme Value Index (EVI).

    Uses Maximum Likelihood Estimation (MLE) to fit the GPD to the excess data
    and returns the shape parameter (EVI/xi).

    Args:
        excess_data: Array of threshold exceedances.

    Returns:
        The estimated Extreme Value Index (shape parameter xi).
    """
    if len(excess_data) < 2:
        return 0.0

    shape, _, _ = genpareto.fit(excess_data, floc=0)
    return float(shape)


def compute_gpd_ci(
    evi: float, confidence_level: float, sample_size: int
) -> Tuple[float, float]:
    """Compute asymptotic confidence interval for the EVI.

    Based on the asymptotic normality of the MLE estimator (Equation 18 in
    computing_confidence_interval.md), the confidence interval is computed
    using the standard error derived from the Cramér-Rao bound.

    The asymptotic distribution is: sqrt(n)(xi_hat - xi) -> N(0, xi^2)
    Therefore, the standard error of xi_hat is |xi|/sqrt(n).

    Args:
        evi: The estimated Extreme Value Index.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        sample_size: Number of observations used in the estimation.

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
    """
    if sample_size < 2:
        return (float("-inf"), float("inf"))

    z_value = norm.ppf((1 + confidence_level) / 2)

    std_error = abs(evi) / np.sqrt(sample_size)

    lower = evi - z_value * std_error
    upper = evi + z_value * std_error

    return (float(lower), float(upper))


def is_in_interval(value: float, interval: Tuple[float, float]) -> bool:
    """Check if a value is within the given interval.

    Args:
        value: The value to check.
        interval: Tuple of (lower_bound, upper_bound).

    Returns:
        True if value is within the interval, False otherwise.
    """
    return interval[0] <= value <= interval[1]
