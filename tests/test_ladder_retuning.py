import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmarlo.replica_exchange.diagnostics import retune_temperature_ladder


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
