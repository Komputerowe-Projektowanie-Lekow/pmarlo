import json
from pathlib import Path

import pytest

from pmarlo.replica_exchange.diagnostics import retune_temperature_ladder
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def test_retune_temperature_ladder(tmp_path):
    temps = [300.0, 330.0, 360.0, 390.0]
    pair_attempt = {(0, 1): 100, (1, 2): 100, (2, 3): 100}
    pair_accept = {(0, 1): 70, (1, 2): 60, (2, 3): 50}
    out_file = tmp_path / "temps.json"

    result = retune_temperature_ladder(
        temps,
        pair_attempt,
        pair_accept,
        target_acceptance=0.30,
        output_json=str(out_file),
        dry_run=True,
    )

    assert out_file.exists()
    suggested = json.loads(out_file.read_text())
    assert len(suggested) <= len(temps)

    expected_global = sum(pair_accept.values()) / sum(pair_attempt.values())
    assert abs(result["global_acceptance"] - expected_global) < 0.02


def test_replica_exchange_tune(tmp_path):
    pdb = str(Path(__file__).parent / "data" / "3gd8.pdb")
    temps = [300.0, 330.0, 360.0, 390.0]
    remd = ReplicaExchange(
        pdb_file=pdb,
        temperatures=temps,
        output_dir=str(tmp_path),
        auto_setup=False,
        target_accept=0.3,
    )
    remd.pair_attempt_counts = {(0, 1): 100, (1, 2): 100, (2, 3): 100}
    remd.pair_accept_counts = {(0, 1): 70, (1, 2): 60, (2, 3): 50}
    new_temps = remd.tune_temperature_ladder()
    assert new_temps[0] == pytest.approx(temps[0])
    assert new_temps[-1] == pytest.approx(temps[-1])
    assert len(new_temps) < len(temps)
    assert remd.n_replicas == len(new_temps)
