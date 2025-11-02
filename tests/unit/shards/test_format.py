import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.shards.format import read_shard_npz_json


def _write_malicious_npz(npz_path: Path) -> None:
    np.savez_compressed(
        npz_path,
        X=np.array([[1.0]], dtype=object),
        t_index=np.array([0], dtype=object),
        dt_ps=np.array(1.0, dtype=object),
    )


def _write_metadata(json_path: Path) -> None:
    payload = {
        "schema_version": "1",
        "shard_id": "T300K_default_seg0000_rep000",
        "temperature_K": 300.0,
        "beta": 1.0,
        "replica_id": 0,
        "segment_id": 0,
        "exchange_window_id": 0,
        "n_frames": 1,
        "dt_ps": 1.0,
        "feature_spec": {
            "name": "test",
            "scaler": "none",
            "columns": ["f0"],
        },
        "provenance": {"kind": "demux"},
    }
    json_path.write_text(json.dumps(payload))


def test_read_shard_npz_json_rejects_object_arrays(tmp_path):
    npz_path = tmp_path / "shard.npz"
    json_path = tmp_path / "shard.json"
    _write_malicious_npz(npz_path)
    _write_metadata(json_path)

    with pytest.raises(ValueError, match="Object arrays"):  # numpy error message
        read_shard_npz_json(npz_path, json_path)
