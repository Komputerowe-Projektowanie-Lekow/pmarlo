import numpy as np
import openmm
import pytest

from pmarlo.replica_exchange.exchange_engine import ExchangeEngine


class _StubRNG:
    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return float(next(self._values))


def test_calculate_probability_matches_metropolis_expression():
    rng = np.random.default_rng(42)
    engine = ExchangeEngine([300.0, 600.0], rng)
    energies = [
        20.0 * openmm.unit.kilojoules_per_mole,
        10.0 * openmm.unit.kilojoules_per_mole,
    ]
    replica_states = [0, 1]

    prob = engine.calculate_probability(replica_states, energies, 0, 1)

    beta_i = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * 300.0 * openmm.unit.kelvin)
    beta_j = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * 600.0 * openmm.unit.kelvin)
    delta_q = (beta_i - beta_j) * (energies[1] - energies[0])
    expected = float(min(1.0, np.exp(delta_q.value_in_unit(openmm.unit.dimensionless))))

    assert prob == pytest.approx(expected)


def test_calculate_probability_is_one_for_favorable_swap():
    rng = np.random.default_rng(7)
    engine = ExchangeEngine([300.0, 600.0], rng)
    energies = [
        5.0 * openmm.unit.kilojoules_per_mole,
        25.0 * openmm.unit.kilojoules_per_mole,
    ]
    replica_states = [0, 1]

    prob = engine.calculate_probability(replica_states, energies, 0, 1)

    assert prob == pytest.approx(1.0)


def test_calculate_probability_requires_openmm_quantities():
    rng = np.random.default_rng(13)
    engine = ExchangeEngine([300.0, 600.0], rng)
    energies = [
        5.0,
        10.0 * openmm.unit.kilojoules_per_mole,
    ]

    with pytest.raises(TypeError):
        engine.calculate_probability([0, 1], energies, 0, 1)


def test_accept_uses_rng_threshold():
    rng = _StubRNG([0.2, 0.8])
    engine = ExchangeEngine([300.0, 600.0], rng)  # temperatures unused here

    assert engine.accept(0.5) is True
    assert engine.accept(0.5) is False
