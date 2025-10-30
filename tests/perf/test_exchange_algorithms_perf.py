# Run with: pytest -m benchmark tests/perf/test_exchange_algorithms_perf.py
"""
Micro-benchmarks for exchange-critical algorithms (energy evaluation and
temperature ladder generation).
"""

import os
import sys
import types
from typing import Iterable

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.replica]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _ensure_openmm_stub() -> None:
    """Provide a lightweight OpenMM stub when the real package is unavailable."""

    try:  # pragma: no cover - exercised only when OpenMM missing
        import openmm  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        unit_stub = types.SimpleNamespace(
            MOLAR_GAS_CONSTANT_R=8.31446261815324,  # J/(mol*K)
            kelvin=1.0,
            dimensionless=1.0,
        )
        openmm_stub = types.ModuleType("openmm")
        openmm_stub.unit = unit_stub
        sys.modules.setdefault("openmm", openmm_stub)
        sys.modules.setdefault("openmm.unit", unit_stub)


class FakeQuantity:
    """Minimal quantity wrapper implementing OpenMM's value API."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def value_in_unit(self, _unit: float) -> float:
        return float(self.value)

    def __sub__(self, other: "FakeQuantity | float") -> "FakeQuantity":
        return FakeQuantity(self.value - _as_float(other))

    def __rsub__(self, other: "FakeQuantity | float") -> "FakeQuantity":
        return FakeQuantity(_as_float(other) - self.value)

    def __mul__(self, other: "FakeQuantity | float") -> "FakeQuantity":
        return FakeQuantity(self.value * _as_float(other))

    __rmul__ = __mul__


def _as_float(value: "FakeQuantity | float") -> float:
    return value.value if isinstance(value, FakeQuantity) else float(value)


def _harmonic_energy(coords: np.ndarray, *, k: float = 1.5) -> float:
    """Return harmonic potential energy (kJ/mol) for the supplied coordinates."""

    diff = coords - np.mean(coords, axis=0, keepdims=True)
    return 0.5 * k * float(np.sum(diff * diff))


def _batched_harmonic_energies(
    positions: Iterable[np.ndarray], *, k: float = 1.5
) -> list[FakeQuantity]:
    return [FakeQuantity(_harmonic_energy(block, k=k)) for block in positions]


def test_exchange_probability_harmonic_energy_benchmark(benchmark) -> None:
    """Benchmark exchange probability calculation using harmonic energies."""

    _ensure_openmm_stub()
    from pmarlo.replica_exchange.exchange_engine import ExchangeEngine

    rng = np.random.default_rng(14)
    n_atoms = 2048
    coords = rng.normal(scale=0.15, size=(2, n_atoms, 3)).astype(np.float64)
    engine = ExchangeEngine([300.0, 450.0], rng)
    replica_states = [0, 1]

    def _compute() -> float:
        energies = _batched_harmonic_energies(coords)
        return engine.calculate_probability(replica_states, energies, 0, 1)

    probability = benchmark(_compute)
    assert 0.0 <= probability <= 1.0


def test_geometric_temperature_ladder_benchmark(benchmark) -> None:
    """Benchmark geometric/exponential temperature ladder construction."""

    from pmarlo.utils.replica_utils import exponential_temperature_ladder

    def _generate() -> list[float]:
        return exponential_temperature_ladder(300.0, 1200.0, 128)

    ladder = benchmark(_generate)
    assert ladder[0] == pytest.approx(300.0)
    assert ladder[-1] == pytest.approx(1200.0)
    assert np.all(np.diff(ladder) > 0.0)


def test_power_two_temperature_ladder_benchmark(benchmark) -> None:
    """Benchmark power-of-two temperature ladder heuristic on large spans."""

    from pmarlo.utils.replica_utils import power_of_two_temperature_ladder

    def _generate() -> list[float]:
        return power_of_two_temperature_ladder(280.0, 1500.0, n_replicas=900)

    ladder = benchmark(_generate)
    assert ladder[0] == pytest.approx(280.0)
    assert ladder[-1] == pytest.approx(1500.0)
    assert len(ladder) & (len(ladder) - 1) == 0  # power of two count
