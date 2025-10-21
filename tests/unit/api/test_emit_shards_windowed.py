from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_API_PATH = _ROOT / "src" / "pmarlo" / "api.py"

with _API_PATH.open("r", encoding="utf-8") as fh:
    _API_SOURCE = fh.read()

_module_ast = ast.parse(_API_SOURCE, filename=str(_API_PATH))
_emit_node = next(
    node
    for node in _module_ast.body
    if isinstance(node, ast.FunctionDef) and node.name == "_emit_windows"
)
_emit_module = ast.Module(body=[_emit_node], type_ignores=[])
_namespace: dict[str, object] = {
    "np": np,
    "Path": Path,
    "Callable": Callable,
    "list": list,
    "dict": dict,
}
exec(compile(_emit_module, str(_API_PATH), "exec"), _namespace)
_emit_windows = _namespace["_emit_windows"]


class FakeSeries:
    def __init__(self, values: list[float]):
        self._values = list(values)
        self.shape = (len(self._values),)

    def __getitem__(self, item):
        return self._values[item]


def _fake_write_shard(
    *, out_dir, shard_id, cvs, dtraj, periodic, seed, temperature, source
):
    path = Path(out_dir) / f"{shard_id}.json"
    path.write_text("{}")
    return path


def test_emit_windows_produces_canonical_metadata(tmp_path):
    rg = FakeSeries([0.1 * i for i in range(10)])
    rmsd = FakeSeries([0.2 * i for i in range(10)])
    provenance = {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "run_id": "run-001",
        "note": "unit-test",
    }

    recorded: list[dict] = []

    def recorder(**kwargs):
        recorded.append(kwargs)
        return _fake_write_shard(**kwargs)

    shard_paths, next_idx = _emit_windows(
        rg,
        rmsd,
        window=5,
        hop=5,
        next_idx=0,
        seed_for=lambda idx: 42 + idx,
        out_dir=tmp_path,
        traj_path=tmp_path / "traj.dcd",
        write_shard=recorder,
        temperature=300.0,
        replica_id=2,
        provenance=provenance,
    )

    assert next_idx == 2
    assert [p.name for p in shard_paths] == [
        "T300K_seg0000_rep002.json",
        "T300K_seg0001_rep002.json",
    ]
    assert recorded[0]["seed"] == 42
    assert recorded[1]["seed"] == 43
    for idx, call in enumerate(recorded):
        source = call["source"]
        assert source["run_id"] == "run-001"
        assert source["kind"] == "demux"
        assert source["created_at"] == "1970-01-01T00:00:00Z"
        assert source["segment_id"] == idx
        assert source["replica_id"] == 2
        assert source["range"] == [idx * 5, (idx + 1) * 5]


def test_emit_windows_requires_mandatory_provenance(tmp_path):
    rg = FakeSeries([0.1 * i for i in range(5)])
    rmsd = FakeSeries([0.2 * i for i in range(5)])

    with pytest.raises(ValueError, match="created_at"):
        _emit_windows(
            rg,
            rmsd,
            window=5,
            hop=5,
            next_idx=0,
            seed_for=lambda idx: idx,
            out_dir=tmp_path,
            traj_path=tmp_path / "traj.dcd",
            write_shard=_fake_write_shard,
            temperature=300.0,
            replica_id=0,
            provenance={"kind": "demux", "run_id": "missing-created"},
        )
