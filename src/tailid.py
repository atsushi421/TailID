"""TailID Algorithm for detecting low-density mixtures in high-quantile tails.

This module implements the main TailID algorithm for detecting ID-sensitive
points in the tail of probability distributions, specifically designed for
probabilistic Worst-Case Execution Time (pWCET) estimation in real-time
systems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from src.data_processing import excess_set, quantile, select_candidates
from src.gpd_statistics import compute_gpd_ci, fit_gpd_evi, is_in_interval

MOS_DEFAULT = 40


class TailIDScenario(Enum):
    """Scenarios for TailID outcomes based on the number of detected points."""

    SCENARIO_1 = 1
    SCENARIO_2 = 2
    SCENARIO_3 = 3


@dataclass
class TailIDResult:
    """Result of the TailID algorithm with scenario interpretation.

    Attributes:
        sensitive_points: List of ID-sensitive points detected.
        scenario: The scenario classification based on |S| and MoS.
        message: Human-readable interpretation of the result.
        tail_threshold: For Scenario 2, the first detected sensitive point
            which becomes the new tail threshold. None for other scenarios.
    """

    sensitive_points: List[float]
    scenario: TailIDScenario
    message: str
    tail_threshold: Optional[float] = None


def _interpret_result(s: List[float], mos: int = MOS_DEFAULT) -> TailIDResult:
    """Interpret TailID results based on the three scenarios from the paper.

    Args:
        s: List of detected sensitive points.
        mos: Minimum of Samples threshold (default: 40).

    Returns:
        TailIDResult with scenario classification and interpretation message.
    """
    num_sensitive = len(s)

    if num_sensitive == 0:
        return TailIDResult(
            sensitive_points=s,
            scenario=TailIDScenario.SCENARIO_1,
            message=(
                "Scenario 1: No inconsistent points detected (|S| = 0). "
                "The identical distribution (ID) hypothesis holds for the "
                "tail. The tail is stable and suitable for pWCET estimation. "
                "Consider combining with KPSS test for additional validation."
            ),
        )
    elif num_sensitive > mos:
        return TailIDResult(
            sensitive_points=s,
            scenario=TailIDScenario.SCENARIO_2,
            message=(
                f"Scenario 2: Many inconsistent points detected "
                f"(|S| = {num_sensitive} > MoS = {mos}). "
                f"Multiple mixture components exist in the tail distribution. "
                f"The first detected point ({s[0]}) becomes the new tail "
                f"threshold, corresponding to the last mixture component. "
                f"Consider this threshold when performing pWCET estimation."
            ),
            tail_threshold=s[0],
        )
    else:
        return TailIDResult(
            sensitive_points=s,
            scenario=TailIDScenario.SCENARIO_3,
            message=(
                f"Scenario 3: Few inconsistent points detected "
                f"(|S| = {num_sensitive} <= MoS = {mos}). "
                f"Mixture distribution exists but insufficient samples for "
                f"accurate parameter estimation. "
                f"Tail prediction should NOT be performed in this state. "
                f"Collect additional samples to reach |S| > MoS for reliable "
                f"estimation."
            ),
        )


def tail_id(
    x: NDArray[np.floating],
    p_m: float,
    p_c1: float,
    gamma: float,
    mos: int = MOS_DEFAULT,
) -> TailIDResult:
    """Detect tail ID-sensitive points using the TailID algorithm.

    This algorithm detects points in the sample data where tail behavior
    shifts, indicating the presence of mixture components. By identifying
    these "ID-sensitive points," the algorithm enables more accurate tail
    modeling and pWCET estimation.

    The result includes scenario classification based on the number of
    detected points (|S|) and the Minimum of Samples (MoS) threshold:
    - Scenario 1 (|S| = 0): No inconsistent points, ID hypothesis holds
    - Scenario 2 (|S| > MoS): Many points detected, use first as threshold
    - Scenario 3 (0 < |S| <= MoS): Few points, insufficient for estimation

    Args:
        x: Sample data for analysis (execution time measurements).
        p_m: Extreme value percentile (defines threshold for tail analysis).
            Must be less than p_c1.
        p_c1: Candidate percentile (defines starting point for candidate set).
            Must be greater than p_m.
        gamma: Confidence level (controls detection sensitivity, e.g., 0.95).
        mos: Minimum of Samples threshold for scenario classification
            (default: 40).

    Returns:
        TailIDResult containing:
        - sensitive_points: List of ID-sensitive points
        - scenario: Classification (SCENARIO_1, SCENARIO_2, or SCENARIO_3)
        - message: Human-readable interpretation and recommended action
        - tail_threshold: For Scenario 2, the first detected point

    Raises:
        ValueError: If p_m >= p_c1 or if parameters are out of valid range.
    """
    if not (0 < p_m < 1):
        raise ValueError("p_m must be between 0 and 1 (exclusive)")
    if not (0 < p_c1 < 1):
        raise ValueError("p_c1 must be between 0 and 1 (exclusive)")
    if not (0 < gamma < 1):
        raise ValueError("gamma must be between 0 and 1 (exclusive)")
    if p_m >= p_c1:
        raise ValueError("p_m must be less than p_c1")

    s: List[float] = []

    t_m = quantile(x, p_m)

    c = select_candidates(x, p_c1)

    if len(c) == 0:
        return _interpret_result(s, mos)

    x_current = x[~np.isin(x, c)]

    y_current = excess_set(x_current, t_m)

    if len(y_current) < 2:
        return _interpret_result(list(c), mos)

    evi_current = fit_gpd_evi(y_current)

    ci_current = compute_gpd_ci(evi_current, gamma, len(y_current))

    for i, c_i in enumerate(c):
        if len(s) == 0:
            x_current = np.append(x_current, c_i)

            y_current = excess_set(x_current, t_m)

            evi_new = fit_gpd_evi(y_current)

            if not is_in_interval(evi_new, ci_current):
                s.append(float(c_i))
            else:
                ci_current = compute_gpd_ci(evi_new, gamma, len(y_current))
        else:
            s.append(float(c_i))

    return _interpret_result(s, mos)
