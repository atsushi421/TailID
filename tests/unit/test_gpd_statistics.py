"""Unit tests for the GPD fitting and statistical functions."""

import numpy as np

from src.gpd_statistics import (
    compute_gpd_ci,
    fit_gpd_evi,
    is_in_interval,
)


class TestFitGPDEVI:
    """Tests for the fit_gpd_evi function."""

    def test_fit_gpd_evi_returns_float(self) -> None:
        """Test that fit_gpd_evi returns a float."""
        data = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
        result = fit_gpd_evi(data)
        assert isinstance(result, float)

    def test_fit_gpd_evi_small_sample(self) -> None:
        """Test fit_gpd_evi with very small sample."""
        data = np.array([1.0])
        result = fit_gpd_evi(data)
        assert result == 0.0

    def test_fit_gpd_evi_exponential_data(self) -> None:
        """Test fit_gpd_evi with exponential-like data (EVI should be near 0)."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=1000)
        result = fit_gpd_evi(data)
        assert -0.5 < result < 0.5


class TestComputeGPDCI:
    """Tests for the compute_gpd_ci function."""

    def test_compute_gpd_ci_returns_tuple(self) -> None:
        """Test that compute_gpd_ci returns a tuple."""
        result = compute_gpd_ci(0.1, 0.95, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_compute_gpd_ci_lower_less_than_upper(self) -> None:
        """Test that lower bound is less than upper bound."""
        lower, upper = compute_gpd_ci(0.1, 0.95, 100)
        assert lower < upper

    def test_compute_gpd_ci_contains_evi(self) -> None:
        """Test that CI contains the EVI estimate."""
        evi = 0.1
        lower, upper = compute_gpd_ci(evi, 0.95, 100)
        assert lower <= evi <= upper

    def test_compute_gpd_ci_wider_with_higher_confidence(self) -> None:
        """Test that higher confidence level gives wider CI."""
        lower_95, upper_95 = compute_gpd_ci(0.1, 0.95, 100)
        lower_99, upper_99 = compute_gpd_ci(0.1, 0.99, 100)
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95

    def test_compute_gpd_ci_narrower_with_larger_sample(self) -> None:
        """Test that larger sample size gives narrower CI."""
        lower_small, upper_small = compute_gpd_ci(0.1, 0.95, 50)
        lower_large, upper_large = compute_gpd_ci(0.1, 0.95, 500)
        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_large < width_small

    def test_compute_gpd_ci_small_sample(self) -> None:
        """Test compute_gpd_ci with very small sample."""
        lower, upper = compute_gpd_ci(0.1, 0.95, 1)
        assert lower == float("-inf")
        assert upper == float("inf")

    def test_compute_gpd_ci_example_15(self) -> None:
        """Test compute_gpd_ci with Example 15 from computing_confidence_interval.md.

        For xi_hat = 1, n = 100, gamma = 0.95:
        CI should be [0.804, 1.196] (approximately).
        """
        lower, upper = compute_gpd_ci(1.0, 0.95, 100)
        assert abs(lower - 0.804) < 0.001
        assert abs(upper - 1.196) < 0.001


class TestIsInInterval:
    """Tests for the is_in_interval function."""

    def test_is_in_interval_inside(self) -> None:
        """Test value inside interval."""
        assert is_in_interval(0.5, (0.0, 1.0)) is True

    def test_is_in_interval_outside_below(self) -> None:
        """Test value below interval."""
        assert is_in_interval(-0.5, (0.0, 1.0)) is False

    def test_is_in_interval_outside_above(self) -> None:
        """Test value above interval."""
        assert is_in_interval(1.5, (0.0, 1.0)) is False

    def test_is_in_interval_on_lower_bound(self) -> None:
        """Test value on lower bound."""
        assert is_in_interval(0.0, (0.0, 1.0)) is True

    def test_is_in_interval_on_upper_bound(self) -> None:
        """Test value on upper bound."""
        assert is_in_interval(1.0, (0.0, 1.0)) is True
