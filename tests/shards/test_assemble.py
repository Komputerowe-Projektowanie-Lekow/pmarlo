from __future__ import annotations

import json
from pathlib import Path

from pmarlo.shards.assemble import select_shards


def _write_shard_json(base_dir: Path, name: str, temperature: float) -> Path:
    payload = {
        "schema_version": "2.0",
        "shard_id": f"{name}-id",
        "temperature_K": temperature,
        "beta": 1.0 / temperature,
        "replica_id": 0,
        "segment_id": 0,
        "exchange_window_id": 0,
        "n_frames": 10,
        "dt_ps": 1.0,
        "feature_spec": {
            "name": "features",
            "scaler": "none",
            "columns": ["c0"],
        },
        "provenance": {"source": "unit-test"},
    }
    json_path = base_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload))
    return json_path


def test_select_shards_requires_exact_temperature_match(tmp_path: Path) -> None:
    cold = _write_shard_json(tmp_path, "cold", 299.999)
    exact = _write_shard_json(tmp_path, "exact", 300.0)
    warm = _write_shard_json(tmp_path, "warm", 300.0005)

    selected = select_shards(tmp_path, temperature_K=300.0)

    assert selected == [exact]
    assert cold not in selected and warm not in selected
