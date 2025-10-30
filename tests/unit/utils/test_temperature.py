from __future__ import annotations

import pytest

from pmarlo.utils.temperature import collect_temperature_values, primary_temperature


def test_collect_temperature_values_top_level_temperature():
    meta = {"temperature": 300}
    temps = collect_temperature_values(meta)
    assert temps == [300.0]


def test_collect_temperature_values_nested_sources():
    meta = {"source": {"metadata": {"temperature_K": "310"}}}
    temps = collect_temperature_values(meta)
    assert temps == [310.0]


def test_collect_temperature_values_sequence_of_dicts():
    meta = {"source": {"metadata": [{"temperature": 310}, {"temperature_K": 320}]}}
    temps = collect_temperature_values(meta)
    assert temps == [310.0, 320.0]


def test_collect_temperature_values_deduplicates_within_tolerance():
    meta = {
        "temperature_K": 300.0,
        "source": {"temperature": 300.00001},
    }
    temps = collect_temperature_values(meta, dedupe_tol=1e-3)
    assert temps == [300.0]


def test_primary_temperature_skips_invalid_values():
    meta = {
        "temperature": None,
        "source": {"temperature_K": "300K"},
    }
    temp = primary_temperature(meta)
    assert temp == pytest.approx(300.0)


def test_primary_temperature_returns_none_when_missing():
    assert primary_temperature({}) is None
    assert primary_temperature(None) is None
