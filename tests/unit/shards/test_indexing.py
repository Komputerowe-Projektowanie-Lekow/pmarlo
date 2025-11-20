from pathlib import Path

import pytest

from pmarlo.shards.indexing import initialise_shard_indices


def test_initialise_shard_indices_counts_canonical_segments(tmp_path: Path) -> None:
    """Only canonical shard filenames should contribute to the next index."""

    canonical = [
        "T300K_run_seg0000_rep000.json",
        "T300K_run_seg0001_rep000.json",
        "T300K_run_seg0001_rep001.json",
    ]
    for name in canonical:
        (tmp_path / name).write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")

    shard_state = initialise_shard_indices(tmp_path)

    assert shard_state.start_index == 2
    assert shard_state.next_index == 2
    assert shard_state.seed_base == 2


def test_initialise_shard_indices_rejects_legacy_pattern(tmp_path: Path) -> None:
    """Legacy shard filenames should fail fast instead of being silently accepted."""

    (tmp_path / "shard_0003.json").write_text("{}")

    with pytest.raises(ValueError) as excinfo:
        initialise_shard_indices(tmp_path)

    message = str(excinfo.value)
    assert "Legacy shard filenames" in message
    assert "shard_0003.json" in message


def test_initialise_shard_indices_requires_numeric_segment(tmp_path: Path) -> None:
    """Malformed canonical names should raise to prevent silent mis-indexing."""

    (tmp_path / "T300K_run_segXXXX_rep000.json").write_text("{}")

    with pytest.raises(ValueError) as excinfo:
        initialise_shard_indices(tmp_path)

    assert "non-numeric segment" in str(excinfo.value)
