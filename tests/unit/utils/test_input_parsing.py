import pytest

from pmarlo.utils.input_parsing import parse_tau_schedule, parse_temperature_ladder


def test_parse_temperature_ladder_accepts_commas_and_semicolons():
    temps = parse_temperature_ladder("300, 320; 340")

    assert temps == [300.0, 320.0, 340.0]


def test_parse_temperature_ladder_from_sequence():
    temps = parse_temperature_ladder([300, 320.0, 340])

    assert temps == [300.0, 320.0, 340.0]


def test_parse_temperature_ladder_rejects_empty_input():
    with pytest.raises(ValueError, match="Provide at least one temperature"):
        parse_temperature_ladder(" , ;  ")


def test_parse_tau_schedule_normalizes_and_sorts_unique_values():
    tau_values = parse_tau_schedule("5, 10; 20, 5")

    assert tau_values == [5, 10, 20]


def test_parse_tau_schedule_rejects_non_positive_values():
    with pytest.raises(ValueError, match="Tau values must be positive integers"):
        parse_tau_schedule("5, 0, -1")
