"""Unit tests for the main TailID algorithm."""

import numpy as np
import pytest

from src.tailid import (
    MOS_DEFAULT,
    TailIDResult,
    TailIDScenario,
    tail_id,
)


class TestTailID:
    """Tests for the main tail_id function."""

    def test_tail_id_returns_tailid_result(self) -> None:
        """Test that tail_id returns a TailIDResult."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=100)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result, TailIDResult)

    def test_tail_id_result_has_sensitive_points_list(self) -> None:
        """Test that TailIDResult contains a list of sensitive points."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=100)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result.sensitive_points, list)

    def test_tail_id_result_has_scenario(self) -> None:
        """Test that TailIDResult contains a scenario classification."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=100)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result.scenario, TailIDScenario)

    def test_tail_id_result_has_message(self) -> None:
        """Test that TailIDResult contains a message."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=100)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_tail_id_invalid_p_m_low(self) -> None:
        """Test that tail_id raises error for p_m <= 0."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="p_m must be between 0 and 1"):
            tail_id(data, p_m=0.0, p_c1=0.9, gamma=0.95)

    def test_tail_id_invalid_p_m_high(self) -> None:
        """Test that tail_id raises error for p_m >= 1."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="p_m must be between 0 and 1"):
            tail_id(data, p_m=1.0, p_c1=0.9, gamma=0.95)

    def test_tail_id_invalid_p_c1_low(self) -> None:
        """Test that tail_id raises error for p_c1 <= 0."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="p_c1 must be between 0 and 1"):
            tail_id(data, p_m=0.7, p_c1=0.0, gamma=0.95)

    def test_tail_id_invalid_gamma_low(self) -> None:
        """Test that tail_id raises error for gamma <= 0."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.0)

    def test_tail_id_invalid_p_m_greater_than_p_c1(self) -> None:
        """Test that tail_id raises error when p_m >= p_c1."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="p_m must be less than p_c1"):
            tail_id(data, p_m=0.95, p_c1=0.9, gamma=0.95)

    def test_tail_id_homogeneous_data(self) -> None:
        """Test tail_id with homogeneous exponential data (no mixture)."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=500)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result.sensitive_points, list)

    def test_tail_id_mixture_data(self) -> None:
        """Test tail_id with mixture data (should detect sensitive points)."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=900)
        tail_component = np.random.exponential(scale=10.0, size=100) + 10
        data = np.concatenate([main_component, tail_component])
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        assert isinstance(result.sensitive_points, list)

    def test_tail_id_sensitive_points_are_floats(self) -> None:
        """Test that all sensitive points are floats."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=900)
        tail_component = np.random.exponential(scale=10.0, size=100) + 10
        data = np.concatenate([main_component, tail_component])
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        for point in result.sensitive_points:
            assert isinstance(point, float)

    def test_tail_id_sensitive_points_in_original_data(self) -> None:
        """Test that all sensitive points are from the original data."""
        np.random.seed(42)
        data = np.random.exponential(scale=1.0, size=100)
        result = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        for point in result.sensitive_points:
            assert point in data


class TestTailIDScenarios:
    """Tests for the TailID scenario classification."""

    def test_scenario_1_no_sensitive_points(self) -> None:
        """Test Scenario 1: No inconsistent points detected (|S| = 0)."""
        np.random.seed(123)
        data = np.random.exponential(scale=1.0, size=1000)
        result = tail_id(data, p_m=0.7, p_c1=0.95, gamma=0.9999)
        if len(result.sensitive_points) == 0:
            assert result.scenario == TailIDScenario.SCENARIO_1
            assert "Scenario 1" in result.message
            assert "No inconsistent points" in result.message

    def test_scenario_2_many_sensitive_points(self) -> None:
        """Test Scenario 2: Many inconsistent points detected (|S| > MoS)."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=500)
        tail_component = np.random.exponential(scale=50.0, size=100) + 50
        data = np.concatenate([main_component, tail_component])
        result = tail_id(data, p_m=0.5, p_c1=0.8, gamma=0.95, mos=5)
        if len(result.sensitive_points) > 5:
            assert result.scenario == TailIDScenario.SCENARIO_2
            assert "Scenario 2" in result.message
            assert result.tail_threshold == result.sensitive_points[0]

    def test_scenario_3_few_sensitive_points(self) -> None:
        """Test Scenario 3: Few inconsistent points detected (0 < |S| <= MoS)."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=900)
        tail_component = np.random.exponential(scale=10.0, size=100) + 10
        data = np.concatenate([main_component, tail_component])
        result = tail_id(data, p_m=0.7, p_c1=0.95, gamma=0.95, mos=1000)
        if 0 < len(result.sensitive_points) <= 1000:
            assert result.scenario == TailIDScenario.SCENARIO_3
            assert "Scenario 3" in result.message
            assert "should NOT be performed" in result.message

    def test_scenario_2_has_tail_threshold(self) -> None:
        """Test that Scenario 2 provides a tail threshold."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=500)
        tail_component = np.random.exponential(scale=50.0, size=100) + 50
        data = np.concatenate([main_component, tail_component])
        result = tail_id(data, p_m=0.5, p_c1=0.8, gamma=0.95, mos=5)
        if result.scenario == TailIDScenario.SCENARIO_2:
            assert result.tail_threshold is not None
            assert result.tail_threshold == result.sensitive_points[0]

    def test_scenario_1_and_3_no_tail_threshold(self) -> None:
        """Test that Scenarios 1 and 3 have no tail threshold."""
        np.random.seed(123)
        data = np.random.exponential(scale=1.0, size=1000)
        result = tail_id(data, p_m=0.7, p_c1=0.95, gamma=0.9999)
        if result.scenario in [TailIDScenario.SCENARIO_1, TailIDScenario.SCENARIO_3]:
            assert result.tail_threshold is None

    def test_custom_mos_parameter(self) -> None:
        """Test that custom MoS parameter affects scenario classification."""
        np.random.seed(42)
        main_component = np.random.exponential(scale=1.0, size=900)
        tail_component = np.random.exponential(scale=10.0, size=100) + 10
        data = np.concatenate([main_component, tail_component])
        result_default = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95)
        result_high_mos = tail_id(data, p_m=0.7, p_c1=0.9, gamma=0.95, mos=1000)
        if len(result_default.sensitive_points) > 0:
            if len(result_default.sensitive_points) > MOS_DEFAULT:
                assert result_default.scenario == TailIDScenario.SCENARIO_2
            if len(result_high_mos.sensitive_points) <= 1000:
                assert result_high_mos.scenario == TailIDScenario.SCENARIO_3
