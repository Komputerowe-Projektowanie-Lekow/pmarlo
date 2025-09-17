"""
Unit tests for temperature ladder utilities.
"""

from __future__ import annotations

import pytest

from pmarlo.utils.replica_utils import (
    exponential_temperature_ladder,
    linear_temperature_ladder,
    power_of_two_temperature_ladder,
)


def is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


class TestPowerOfTwoTemperatureLadder:
    def test_rounds_up_to_power_of_two(self):
        temps = power_of_two_temperature_ladder(300.0, 375.0, 13)
        assert is_power_of_two(len(temps))
        assert len(temps) == 16

    def test_none_replicas_picks_reasonable_spacing(self):
        temps = power_of_two_temperature_ladder(300.0, 375.0, None)
        assert is_power_of_two(len(temps))
        # Roughly 5K spacing → about 16 points from 300 to 375 inclusive
        assert 8 <= len(temps) <= 32
        assert abs(temps[0] - 300.0) < 1e-8
        assert abs(temps[-1] - 375.0) < 1e-8

    def test_inclusive_bounds_and_sorted(self):
        temps = power_of_two_temperature_ladder(300.0, 375.0, 16)
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(375.0)
        assert temps == sorted(temps)

    def test_handles_min_greater_than_max(self):
        temps = power_of_two_temperature_ladder(375.0, 300.0, 8)
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(375.0)
        assert len(temps) == 8

    def test_degenerate_equal_bounds(self):
        temps = power_of_two_temperature_ladder(310.0, 310.0, None)
        assert temps == [310.0]


class TestOtherLadders:
    def test_linear_ladder_count_and_bounds(self):
        temps = linear_temperature_ladder(300.0, 360.0, 4)
        assert len(temps) == 4
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(360.0)

    def test_linear_ladder_handles_swapped_bounds(self):
        temps = linear_temperature_ladder(360.0, 300.0, 4)
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(360.0)

    def test_exponential_ladder_count_and_bounds(self):
        temps = exponential_temperature_ladder(300.0, 360.0, 5)
        assert len(temps) == 5
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(360.0)

    def test_exponential_ladder_handles_swapped_bounds(self):
        temps = exponential_temperature_ladder(360.0, 300.0, 5)
        assert temps[0] == pytest.approx(300.0)
        assert temps[-1] == pytest.approx(360.0)

    @pytest.mark.parametrize(
        "func",
        [
            linear_temperature_ladder,
            exponential_temperature_ladder,
            power_of_two_temperature_ladder,
        ],
    )
    def test_invalid_n_replicas_raises(self, func):
        with pytest.raises(ValueError):
            func(300.0, 360.0, 0)
