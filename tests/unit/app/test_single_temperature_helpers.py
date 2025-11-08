from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo_webapp.app.backend.sampling import _resolve_primary_temperature
from pmarlo_webapp.app.backend.types import SimulationConfig

PDB_PATH = Path("tests/_assets/3gd8-fixed.pdb")


def _make_config(**overrides: object) -> SimulationConfig:
    payload: dict[str, object] = {
        "pdb_path": overrides.pop("pdb_path", PDB_PATH),
        "temperatures": overrides.pop("temperatures", [310.0]),
        "steps": overrides.pop("steps", 10_000),
        "quick": overrides.pop("quick", True),
    }
    payload.update(overrides)
    return SimulationConfig(**payload)  # type: ignore[arg-type]


def test_simulation_config_snapshot_roundtrip_preserves_single_temperature_flag() -> None:
    config = _make_config(single_temperature_mode=True)
    snapshot = config.snapshot()
    restored = SimulationConfig.from_snapshot(snapshot)
    assert restored.single_temperature_mode is True


@pytest.mark.parametrize(
    "temperatures,restart_temperature,expected",
    [
        ([330.0, 360.0], None, 330.0),
        ([], 315.0, 315.0),
        ([], None, 300.0),
    ],
)
def test_resolve_primary_temperature(temperatures: list[float], restart_temperature: float | None, expected: float) -> None:
    config = _make_config(
        temperatures=temperatures,
        restart_temperature=restart_temperature,
    )
    assert _resolve_primary_temperature(config) == pytest.approx(expected)
