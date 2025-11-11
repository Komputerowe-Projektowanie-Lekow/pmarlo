from __future__ import annotations

from pathlib import Path

from pmarlo.replica_exchange.diagnostics import retune_temperature_ladder


def test_acceptance_global_aggregation(tmp_path: Path):
    temps = [300.0, 310.0, 320.0]
    pair_attempts = {(0, 1): 100, (1, 2): 100}
    pair_accepts = {(0, 1): 30, (1, 2): 20}
    out = tmp_path / "suggested.json"
    res = retune_temperature_ladder(
        temps,
        pair_attempts,
        pair_accepts,
        target_acceptance=0.3,
        output_json=str(out),
        dry_run=True,
    )
    assert abs(res["global_acceptance"] - (50 / 200)) < 1e-9
