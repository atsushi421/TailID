"""TailID package for detecting low-density mixtures in high-quantile tails."""

from src.data_processing import (
    excess_set,
    quantile,
    select_candidates,
)
from src.gpd_statistics import (
    compute_gpd_ci,
    fit_gpd_evi,
    is_in_interval,
)
from src.tailid import (
    MOS_DEFAULT,
    TailIDResult,
    TailIDScenario,
    tail_id,
)
from src.threshold_selection import select_threshold

__all__ = [
    "quantile",
    "select_candidates",
    "excess_set",
    "fit_gpd_evi",
    "compute_gpd_ci",
    "is_in_interval",
    "tail_id",
    "TailIDResult",
    "TailIDScenario",
    "MOS_DEFAULT",
    "select_threshold",
]
