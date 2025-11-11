from __future__ import annotations

import subprocess
import sys
from itertools import count
from pathlib import Path

from pmarlo.data.shard import write_shard
from pmarlo.utils.path_utils import ensure_directory

_SEGMENT_COUNTER = count()


def _canonical_shard_id(temperature_K: float, segment_id: int, replica_id: int) -> str:
    temp = int(round(temperature_K))
    return f"T{temp}K_seg{segment_id:04d}_rep{replica_id:03d}"


def _source_metadata(segment_id: int, replica_id: int) -> dict[str, object]:
    return {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "run_id": "cli-test",
        "segment_id": int(segment_id),
        "replica_id": int(replica_id),
        "exchange_window_id": 0,
    }


def _prepare_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace_dir = tmp_path / "workspace"
    shards_dir = workspace_dir / "shards"
    ensure_directory(workspace_dir / "sims")
    ensure_directory(shards_dir)
    (workspace_dir / "models").mkdir(exist_ok=True)
    (workspace_dir / "bundles").mkdir(exist_ok=True)
    (workspace_dir / "logs").mkdir(exist_ok=True)

    import numpy as np

    rng = np.random.default_rng(99)
    frames = 20000
    cv1 = np.cumsum(rng.normal(size=frames))
    cv2 = np.cumsum(rng.normal(size=frames))
    state = np.zeros(frames, dtype=np.int32)
    for t in range(1, frames):
        if rng.random() < 0.75:
            state[t] = state[t - 1]
        else:
            state[t] = rng.integers(0, 10)

    segment_id = next(_SEGMENT_COUNTER)
    replica_id = 0
    shard_id = _canonical_shard_id(300.0, segment_id, replica_id)
    shard_path = write_shard(
        out_dir=shards_dir,
        shard_id=shard_id,
        cvs={"cv1": cv1, "cv2": cv2},
        dtraj=state,
        periodic={"cv1": False, "cv2": False},
        seed=2025,
        temperature=300.0,
        source=_source_metadata(segment_id, replica_id),
    )
    return workspace_dir, Path(shard_path)


def test_cli_outputs_summary(tmp_path):
    workspace_dir, shard_path = _prepare_workspace(tmp_path)
    state_path = workspace_dir / "state.json"
    state_path.write_text("{}", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "pmarlo_webapp.app.headless",
        "--workspace",
        str(workspace_dir),
        "run-analysis",
        "--shard",
        str(shard_path),
        "--lag",
        "3000",
        "--microstates",
        "40",
        "--cluster-mode",
        "kmeans",
        "--seed",
        "2025",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "tau (frames)" in stdout
    assert "total_pairs" in stdout
