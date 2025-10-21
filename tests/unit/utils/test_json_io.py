from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmarlo.utils.json_io import load_json_file


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
