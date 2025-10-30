from __future__ import annotations

import pytest

from pmarlo.demultiplexing.exchange_validation import normalize_exchange_mapping


def test_normalize_exchange_mapping_accepts_valid_values() -> None:
    mapping = ["0", 1, 2]

    assert normalize_exchange_mapping(mapping, expected_size=3, context="slice=5") == [
        0,
        1,
        2,
    ]


def test_normalize_exchange_mapping_rejects_non_integer_entries() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_exchange_mapping(["a", 1], expected_size=2)

    assert "non-integer" in str(exc_info.value)


def test_normalize_exchange_mapping_rejects_out_of_range_indices() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_exchange_mapping([0, 3], expected_size=2, context="segment 2")

    message = str(exc_info.value)
    assert "out of range" in message
    assert "segment 2" in message


def test_normalize_exchange_mapping_requires_permutation() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_exchange_mapping([0, 0, 1], expected_size=3)

    assert "permutation" in str(exc_info.value)


def test_normalize_exchange_mapping_honours_custom_error_class() -> None:
    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        normalize_exchange_mapping([0, 1], expected_size=3, error_cls=CustomError)
