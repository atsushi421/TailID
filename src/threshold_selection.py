"""Threshold selection for POT approach using EQMAE minimization.

This module implements automatic threshold selection for the Peak Over
Threshold (POT) approach by minimizing the Estimated Quantile Mean Absolute
Error (EQMAE) across candidate thresholds.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto

from src.data_processing import excess_set, quantile

P_M_MIN = 0.6
P_M_MAX = 0.9


def _compute_eqmae(
    data: NDArray[np.floating], threshold: float
) -> float:
    """Compute EQMAE for a given threshold.

    EQMAE (Estimated Quantile Mean Absolute Error) measures how well the
    fitted GP model approximates the observed exceedances.

    EQMAE = (1/n) * sum(|x_hat_i - x_i|)

    where x_i are the observed exceedances and x_hat_i are the quantiles
    estimated from the fitted GP model.

    Args:
        data: Sample data array.
        threshold: Threshold value for computing excesses.

    Returns:
        The EQMAE value. Returns infinity if fitting fails.
    """
    excesses = excess_set(data, threshold)
    n = len(excesses)

    if n < 2:
        return float("inf")

    try:
        shape, loc, scale = genpareto.fit(excesses, floc=0)
    except Exception:
        return float("inf")

    if scale <= 0:
        return float("inf")

    sorted_excesses = np.sort(excesses)

    probabilities = (np.arange(1, n + 1) - 0.5) / n

    estimated_quantiles = genpareto.ppf(
        probabilities, shape, loc=0, scale=scale
    )

    eqmae = np.mean(np.abs(estimated_quantiles - sorted_excesses))

    return float(eqmae)


def select_threshold(
    data: NDArray[np.floating],
    n_candidates: int,
    p_min: float = P_M_MIN,
    p_max: float = P_M_MAX,
) -> float:
    """Select optimal threshold percentile by minimizing EQMAE.

    This function evaluates candidate thresholds between p_min and p_max
    percentiles and selects the one that minimizes the Estimated Quantile
    Mean Absolute Error (EQMAE) when fitting a GP model to the exceedances.

    Based on the approach described in the literature, candidate thresholds
    are restricted to the 60% to 90% percentile range to:
    1. Focus the search on the tail region
    2. Avoid selecting thresholds that are too high

    Args:
        data: Sample data array.
        n_candidates: Number of candidate thresholds to evaluate.
        p_min: Minimum percentile for candidate thresholds (default: 0.6).
        p_max: Maximum percentile for candidate thresholds (default: 0.9).

    Returns:
        The optimal threshold percentile (p_m) that minimizes EQMAE.

    Raises:
        ValueError: If p_min >= p_max or if parameters are out of valid range.
    """
    if not (0 < p_min < 1):
        raise ValueError("p_min must be between 0 and 1 (exclusive)")
    if not (0 < p_max < 1):
        raise ValueError("p_max must be between 0 and 1 (exclusive)")
    if p_min >= p_max:
        raise ValueError("p_min must be less than p_max")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")

    candidate_percentiles = np.linspace(p_min, p_max, n_candidates)

    best_percentile = p_min
    best_eqmae = float("inf")

    for p in candidate_percentiles:
        threshold = quantile(data, p)
        eqmae = _compute_eqmae(data, threshold)

        if eqmae < best_eqmae:
            best_eqmae = eqmae
            best_percentile = p

    return float(best_percentile)
