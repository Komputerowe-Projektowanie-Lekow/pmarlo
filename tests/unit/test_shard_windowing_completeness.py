"""Test that windowed shard emission captures all frames from trajectories."""

import numpy as np
import pytest
from pathlib import Path

from pmarlo.api.shards import _emit_windows


def test_emit_windows_captures_all_frames():
    """Verify that all frames are captured, including partial shards at the end."""
    # Simulate a trajectory with 1950 frames (like in the logs)
    n_frames = 1950
    rg = np.random.uniform(1.0, 2.0, n_frames)
    rmsd = np.random.uniform(0.5, 1.5, n_frames)

    window = 1000
    hop = 1000  # Non-overlapping

    # Mock seed function
    def seed_for(idx):
        return 42 + idx

    # Track emitted shards
    emitted_shards = []

    def mock_write_shard(out_dir, shard_id, cvs, dtraj, periodic, seed, temperature, source):
        emitted_shards.append({
            "shard_id": shard_id,
            "n_frames": cvs["Rg"].shape[0],
            "range": source["range"],
            "partial": source.get("partial", False),
            "is_final_partial": source.get("is_final_partial", False),
        })
        return Path(f"shard_{len(emitted_shards)}.json")

    provenance = {
        "created_at": "2025-11-01",
        "kind": "demux",
        "run_id": "test_run",
    }

    shard_paths, next_idx = _emit_windows(
        rg=rg,
        rmsd=rmsd,
        window=window,
        hop=hop,
        next_idx=0,
        seed_for=seed_for,
        out_dir=Path("/tmp/test"),
        traj_path=Path("test.dcd"),
        write_shard=mock_write_shard,
        temperature=300.0,
        replica_id=0,
        provenance=provenance,
    )

    # Should have 2 shards: one full (1000 frames) + one partial (950 frames)
    assert len(emitted_shards) == 2, f"Expected 2 shards, got {len(emitted_shards)}"

    # First shard: full window
    assert emitted_shards[0]["n_frames"] == 1000
    assert emitted_shards[0]["range"] == [0, 1000]
    assert not emitted_shards[0]["partial"]

    # Second shard: partial (remaining 950 frames)
    assert emitted_shards[1]["n_frames"] == 950
    assert emitted_shards[1]["range"] == [1000, 1950]
    assert emitted_shards[1]["partial"], "Second shard should be marked as partial"
    assert emitted_shards[1]["is_final_partial"], "Second shard should be marked as final partial"

    # Both shards should use canonical ID format (no _partial suffix)
    for shard in emitted_shards:
        shard_id = shard["shard_id"]
        assert "_partial" not in shard_id, f"Shard ID should not contain '_partial' suffix: {shard_id}"
        # Verify canonical format: T{temp}K_{run}_seg{segment:04d}_rep{replica:03d}
        assert shard_id.startswith("T300K_test_run_seg"), f"Unexpected shard ID format: {shard_id}"

    # Verify ALL frames are captured
    total_frames_captured = sum(s["n_frames"] for s in emitted_shards)
    assert total_frames_captured == n_frames, \
        f"Lost {n_frames - total_frames_captured} frames!"


def test_emit_windows_exact_multiple():
    """Test case where trajectory length is exact multiple of window size."""
    n_frames = 2000  # Exactly 2x window size
    rg = np.random.uniform(1.0, 2.0, n_frames)
    rmsd = np.random.uniform(0.5, 1.5, n_frames)

    window = 1000
    hop = 1000

    emitted_shards = []

    def mock_write_shard(out_dir, shard_id, cvs, dtraj, periodic, seed, temperature, source):
        emitted_shards.append({
            "n_frames": cvs["Rg"].shape[0],
            "partial": source.get("partial", False),
        })
        return Path(f"shard_{len(emitted_shards)}.json")

    def seed_for(idx):
        return 42 + idx

    provenance = {
        "created_at": "2025-11-01",
        "kind": "demux",
        "run_id": "test_run",
    }

    shard_paths, next_idx = _emit_windows(
        rg=rg,
        rmsd=rmsd,
        window=window,
        hop=hop,
        next_idx=0,
        seed_for=seed_for,
        out_dir=Path("/tmp/test"),
        traj_path=Path("test.dcd"),
        write_shard=mock_write_shard,
        temperature=300.0,
        replica_id=0,
        provenance=provenance,
    )

    # Should have exactly 2 full shards, no partial
    assert len(emitted_shards) == 2
    assert sum(s["n_frames"] for s in emitted_shards) == n_frames
    assert not any(s["partial"] for s in emitted_shards), "No shards should be partial"


def test_emit_windows_overlapping():
    """Test overlapping windows still capture all frames."""
    n_frames = 1950
    rg = np.random.uniform(1.0, 2.0, n_frames)
    rmsd = np.random.uniform(0.5, 1.5, n_frames)

    window = 1000
    hop = 500  # 50% overlap

    emitted_shards = []

    def mock_write_shard(out_dir, shard_id, cvs, dtraj, periodic, seed, temperature, source):
        emitted_shards.append({
            "n_frames": cvs["Rg"].shape[0],
            "range": source["range"],
        })
        return Path(f"shard_{len(emitted_shards)}.json")

    def seed_for(idx):
        return 42 + idx

    provenance = {
        "created_at": "2025-11-01",
        "kind": "demux",
        "run_id": "test_run",
    }

    shard_paths, next_idx = _emit_windows(
        rg=rg,
        rmsd=rmsd,
        window=window,
        hop=hop,
        next_idx=0,
        seed_for=seed_for,
        out_dir=Path("/tmp/test"),
        traj_path=Path("test.dcd"),
        write_shard=mock_write_shard,
        temperature=300.0,
        replica_id=0,
        provenance=provenance,
    )

    # With hop=500, we should get: 0-1000, 500-1500, 1000-2000 (partial)
    # The last regular window ends at 1500, leaving 450 frames for partial
    assert len(emitted_shards) >= 3

    # Last frame index covered
    max_frame_covered = max(s["range"][1] for s in emitted_shards)
    assert max_frame_covered == n_frames, \
        f"Not all frames covered! Max: {max_frame_covered}, Total: {n_frames}"
