"""Tests for pmarlo.utils.logging_utils."""

import pytest

from pmarlo.utils.logging_utils import format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-5.0, "0 milliseconds"),
        (0.5, "500 milliseconds"),
        (1.234, "1.23 seconds"),
        (59.9, "59.90 seconds"),
        (61.789, "1 minute and 1.8 seconds"),
        (3661.2, "1 hour, 1 minute and 1.2 seconds"),
        (90061.0, "1 day, 1 hour and 1 minute"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """format_duration renders human-readable durations consistently."""

    assert format_duration(seconds) == expected
