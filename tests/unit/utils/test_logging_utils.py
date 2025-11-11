"""Tests for pmarlo.utils.logging_utils."""

import pytest

from pmarlo.utils.logging_utils import format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-5.0, "0 ms"),
        (0.5, "500 ms"),
        (1.234, "1.23 s"),
        (59.9, "59.90 s"),
        (61.789, "1 min 1.8 s"),
        (3661.2, "1 h 1 min 1.2 s"),
        (90061.0, "1 d 1 h 1 min"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """format_duration renders human-readable durations consistently."""

    assert format_duration(seconds) == expected
