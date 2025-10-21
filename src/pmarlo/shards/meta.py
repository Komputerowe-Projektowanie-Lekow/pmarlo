from __future__ import annotations

"""Strict JSON metadata loading for shards."""

from pathlib import Path

from pmarlo.utils.json_io import load_json_file

from .schema import FeatureSpec, ShardMeta

__all__ = ["load_shard_meta"]

# Included in the agents.md


def load_shard_meta(json_path: Path) -> ShardMeta:
    """Load shard metadata strictly with no filename heuristics."""

    payload = load_json_file(json_path)
    feature_spec = FeatureSpec(
        name=str(payload["feature_spec"]["name"]),
        scaler=str(payload["feature_spec"]["scaler"]),
        columns=tuple(payload["feature_spec"]["columns"]),
    )
    return ShardMeta(
        schema_version=str(payload["schema_version"]),
        shard_id=str(payload["shard_id"]),
        temperature_K=float(payload["temperature_K"]),
        beta=float(payload["beta"]),
        replica_id=int(payload["replica_id"]),
        segment_id=int(payload["segment_id"]),
        exchange_window_id=int(payload["exchange_window_id"]),
        n_frames=int(payload["n_frames"]),
        dt_ps=float(payload["dt_ps"]),
        feature_spec=feature_spec,
        provenance=dict(payload["provenance"]),
    )
