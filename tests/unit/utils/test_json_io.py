from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.utils.json_io import load_json_file, sanitize, write_json


def test_load_json_file_reads_payload(tmp_path: Path) -> None:
    path = tmp_path / "shard_meta.json"
    path.write_text('{"schema_version": "1.0", "value": 42}')

    data = load_json_file(path)

    assert data == {"schema_version": "1.0", "value": 42}


def test_load_json_file_annotates_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": ')

    with pytest.raises(json.JSONDecodeError) as excinfo:
        load_json_file(path)

    assert str(path) in str(excinfo.value)


def test_sanitize_handles_numpy_paths_and_fallbacks(tmp_path: Path) -> None:
    class CustomObject:
        def __str__(self) -> str:
            return "custom-object"

    payload = {
        "path": tmp_path / "dataset",
        "array": np.array([[1, 2], [3, 4]]),
        "scalar": np.float64(7.0),
        "non_finite": float("nan"),
        "mapping": {tmp_path: np.int64(5)},
        "sequence": (1, 2, 3),
        "set_like": {3, 1},
        "custom": CustomObject(),
    }

    sanitized = sanitize(payload)

    assert sanitized["path"] == str(tmp_path / "dataset")
    assert sanitized["array"] == [[1, 2], [3, 4]]
    assert sanitized["scalar"] == 7.0
    assert sanitized["non_finite"] is None
    assert sanitized["mapping"][str(tmp_path)] == 5
    assert sanitized["sequence"] == [1, 2, 3]
    assert sorted(sanitized["set_like"]) == [1, 3]
    assert sanitized["custom"] == "custom-object"


def test_write_json_serializes_with_pydantic(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path / "target",
        "values": np.array([1, 2, 3]),
        "bad_float": float("inf"),
    }
    output = tmp_path / "result" / "summary.json"

    write_json(output, payload)

    with output.open() as handle:
        data = json.load(handle)

    assert data == {
        "bad_float": None,
        "path": str(tmp_path / "target"),
        "values": [1, 2, 3],
    }
