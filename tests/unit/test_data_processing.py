"""Unit tests for the data processing utilities."""

import numpy as np

from src.data_processing import (
    excess_set,
    quantile,
    select_candidates,
)


class TestQuantile:
    """Tests for the quantile function."""

    def test_quantile_median(self) -> None:
        """Test that quantile returns correct median."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quantile(data, 0.5)
        assert result == 3.0

    def test_quantile_minimum(self) -> None:
        """Test that quantile at 0 returns minimum."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quantile(data, 0.0)
        assert result == 1.0

    def test_quantile_maximum(self) -> None:
        """Test that quantile at 1 returns maximum."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quantile(data, 1.0)
        assert result == 5.0

    def test_quantile_returns_float(self) -> None:
        """Test that quantile returns a float."""
        data = np.array([1.0, 2.0, 3.0])
        result = quantile(data, 0.5)
        assert isinstance(result, float)


class TestSelectCandidates:
    """Tests for the select_candidates function."""

    def test_select_candidates_basic(self) -> None:
        """Test basic candidate selection."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        candidates = select_candidates(data, 0.8)
        assert len(candidates) >= 2
        assert all(c >= quantile(data, 0.8) for c in candidates)

    def test_select_candidates_sorted(self) -> None:
        """Test that candidates are sorted in ascending order."""
        data = np.array([10.0, 1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0])
        candidates = select_candidates(data, 0.7)
        for i in range(len(candidates) - 1):
            assert candidates[i] <= candidates[i + 1]

    def test_select_candidates_empty_high_percentile(self) -> None:
        """Test candidate selection with very high percentile."""
        data = np.array([1.0, 2.0, 3.0])
        candidates = select_candidates(data, 0.99)
        assert len(candidates) >= 1


class TestExcessSet:
    """Tests for the excess_set function."""

    def test_excess_set_basic(self) -> None:
        """Test basic excess set computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = 3.0
        excesses = excess_set(data, threshold)
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(excesses, expected)

    def test_excess_set_no_exceedances(self) -> None:
        """Test excess set when no values exceed threshold."""
        data = np.array([1.0, 2.0, 3.0])
        threshold = 5.0
        excesses = excess_set(data, threshold)
        assert len(excesses) == 0

    def test_excess_set_all_exceed(self) -> None:
        """Test excess set when all values exceed threshold."""
        data = np.array([5.0, 6.0, 7.0])
        threshold = 4.0
        excesses = excess_set(data, threshold)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(excesses, expected)

    def test_excess_set_excludes_threshold(self) -> None:
        """Test that values equal to threshold are excluded."""
        data = np.array([3.0, 4.0, 5.0])
        threshold = 3.0
        excesses = excess_set(data, threshold)
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(excesses, expected)
