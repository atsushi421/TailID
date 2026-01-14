"""Unit tests for threshold selection module."""

import numpy as np
import pytest

from src.threshold_selection import (
    P_M_MAX,
    P_M_MIN,
    _compute_eqmae,
    select_threshold,
)


class TestComputeEQMAE:
    """Tests for _compute_eqmae function."""

    def test_compute_eqmae_returns_float(self):
        """Test that _compute_eqmae returns a float."""
        data = np.random.exponential(scale=1.0, size=100)
        threshold = np.quantile(data, 0.7)
        result = _compute_eqmae(data, threshold)
        assert isinstance(result, float)

    def test_compute_eqmae_insufficient_data(self):
        """Test that _compute_eqmae returns inf for insufficient data."""
        data = np.array([1.0, 2.0, 3.0])
        threshold = 2.5
        result = _compute_eqmae(data, threshold)
        assert result == float("inf")

    def test_compute_eqmae_no_exceedances(self):
        """Test that _compute_eqmae returns inf when no exceedances."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = 10.0
        result = _compute_eqmae(data, threshold)
        assert result == float("inf")

    def test_compute_eqmae_positive_value(self):
        """Test that _compute_eqmae returns a non-negative value."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=200)
        threshold = np.quantile(data, 0.7)
        result = _compute_eqmae(data, threshold)
        assert result >= 0


class TestSelectThreshold:
    """Tests for select_threshold function."""

    def test_select_threshold_returns_float(self):
        """Test that select_threshold returns a float."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=200)
        result = select_threshold(data, n_candidates=31)
        assert isinstance(result, float)

    def test_select_threshold_within_range(self):
        """Test that selected threshold is within the default range."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=200)
        result = select_threshold(data, n_candidates=31)
        assert P_M_MIN <= result <= P_M_MAX

    def test_select_threshold_custom_range(self):
        """Test that selected threshold is within custom range."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=200)
        p_min, p_max = 0.7, 0.85
        result = select_threshold(data, n_candidates=31, p_min=p_min, p_max=p_max)
        assert p_min <= result <= p_max

    def test_select_threshold_invalid_p_min_low(self):
        """Test that select_threshold raises error for p_min <= 0."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="p_min must be between"):
            select_threshold(data, n_candidates=31, p_min=0)

    def test_select_threshold_invalid_p_min_high(self):
        """Test that select_threshold raises error for p_min >= 1."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="p_min must be between"):
            select_threshold(data, n_candidates=31, p_min=1)

    def test_select_threshold_invalid_p_max_low(self):
        """Test that select_threshold raises error for p_max <= 0."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="p_max must be between"):
            select_threshold(data, n_candidates=31, p_max=0)

    def test_select_threshold_invalid_p_max_high(self):
        """Test that select_threshold raises error for p_max >= 1."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="p_max must be between"):
            select_threshold(data, n_candidates=31, p_max=1)

    def test_select_threshold_p_min_greater_than_p_max(self):
        """Test that select_threshold raises error when p_min >= p_max."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="p_min must be less than p_max"):
            select_threshold(data, n_candidates=31, p_min=0.9, p_max=0.6)

    def test_select_threshold_invalid_n_candidates(self):
        """Test that select_threshold raises error for n_candidates < 2."""
        data = np.random.exponential(scale=1.0, size=100)
        with pytest.raises(ValueError, match="n_candidates must be at least 2"):
            select_threshold(data, n_candidates=1)

    def test_select_threshold_deterministic(self):
        """Test that select_threshold is deterministic for same data."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=200)
        result1 = select_threshold(data, n_candidates=31)
        result2 = select_threshold(data, n_candidates=31)
        assert result1 == result2
