"""Tests for :mod:`pmarlo.utils.validation`."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _load_validation_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "pmarlo"
        / "utils"
        / "validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pmarlo.utils.validation", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load validation module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load_validation_module()
finite_or_none = validation.finite_or_none


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.234, 1.234),
        ("2.5", 2.5),
        (0, 0.0),
    ],
)
def test_finite_or_none_returns_float_for_finite_values(value, expected) -> None:
    result = finite_or_none(value)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [None, math.nan, math.inf, -math.inf, "not-a-number"],
)
def test_finite_or_none_returns_none_for_invalid_inputs(value) -> None:
    assert finite_or_none(value) is None
