"""Behavioral tests for `_find_transition_states`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest

from pmarlo.conformations import finder as m


@dataclass
class DummyTPTResult:
    """Minimal TPT result stand-in for behavioral tests."""

    source_states: np.ndarray
    sink_states: np.ndarray
    forward_committor: np.ndarray
    flux_matrix: Any


@dataclass
class DummyKISResult:
    """Minimal KIS result stand-in for behavioral tests."""

    kis_scores: np.ndarray


def _make_basic_tpt(
    pi: np.ndarray,
    forward_committor: np.ndarray,
    source_states=(0,),
    sink_states=(3,),
) -> DummyTPTResult:
    """Construct a simple TPTResult-like object with aligned lengths."""
    pi = np.asarray(pi, dtype=float)
    fc = np.asarray(forward_committor, dtype=float)
    assert (
        pi.shape[0] == fc.shape[0]
    ), "pi and forward_committor must have same length in tests"
    n = pi.shape[0]
    flux_matrix = np.zeros((n, n), dtype=float)
    return DummyTPTResult(
        source_states=np.array(source_states, dtype=int),
        sink_states=np.array(sink_states, dtype=int),
        forward_committor=fc,
        flux_matrix=flux_matrix,
    )


def _by_state_id(conformations: List[m.Conformation]) -> Dict[int, m.Conformation]:
    """Build a lookup of Conformation objects by state id."""
    return {c.state_id: c for c in conformations}


def test_reactive_states_only_source_and_sink_skipped():
    """All non-source / non-sink states are returned, sources and sinks are excluded."""
    pi = np.array([0.2, 0.3, 0.1, 0.4])
    forward_committor = np.array([0.0, 0.2, 0.8, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(3,))
    flux_by_state = np.array([0.0, 1.0, 2.0, 3.0])

    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.05,
    )

    ids = sorted(c.state_id for c in confs)
    assert ids == [1, 2]
    assert all(c.state_id not in {0, 3} for c in confs)
    assert all(c.conformation_type in {"transition", "tse"} for c in confs)


def test_tse_classification_depends_on_committor_and_tolerance():
    """States within tolerance of 0.5 committor are labeled as TSE."""
    pi = np.array([0.1, 0.3, 0.3, 0.3])
    forward_committor = np.array([0.0, 0.49, 0.50, 0.51])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(3,))
    flux_by_state = np.array([0.0, 1.0, 1.0, 0.0])

    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.01,
    )
    by_id = _by_state_id(confs)
    assert by_id[1].conformation_type == "tse"
    assert by_id[2].conformation_type == "tse"

    confs_tight = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.0,
    )
    by_id_tight = _by_state_id(confs_tight)
    assert by_id_tight[1].conformation_type == "transition"
    assert by_id_tight[2].conformation_type == "tse"


def test_free_energy_comes_from_population():
    """Free energies follow the Boltzmann relation and decrease with population."""
    pi = np.array([0.1, 0.2, 0.4, 0.3])
    forward_committor = np.array([0.0, 0.2, 0.8, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(), sink_states=())
    flux_by_state = np.zeros_like(pi)

    T = 310.0
    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=T,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.05,
    )
    by_id = _by_state_id(confs)

    kT = m.constants.k * T * m.constants.Avogadro / 1000.0
    expected_fe = -kT * np.log(pi)

    for state_id, expected in enumerate(expected_fe):
        assert np.isclose(by_id[state_id].free_energy, expected, rtol=1e-6, atol=1e-6)

    assert by_id[2].free_energy < by_id[1].free_energy < by_id[0].free_energy


def test_free_energy_handles_zero_population_safely():
    """States with zero population produce finite but high free energy."""
    pi = np.array([0.0, 1e-4, 1e-2])
    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(), sink_states=())
    flux_by_state = np.zeros_like(pi)

    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.05,
    )
    by_id = _by_state_id(confs)

    assert np.isfinite(by_id[0].free_energy)
    assert by_id[0].free_energy > by_id[1].free_energy
    assert by_id[0].free_energy > by_id[2].free_energy


def test_flux_and_kis_scores_are_mapped_by_state_index():
    """Flux and KIS scores propagate by state index."""
    pi = np.array([0.2, 0.3, 0.5, 0.0])
    forward_committor = np.array([0.0, 0.3, 0.7, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(3,))
    flux_by_state = np.array([0.0, 10.0, 20.0, 30.0])
    kis_scores = np.array([1.0, 2.0, 3.0, 4.0])
    kis_result = DummyKISResult(kis_scores=kis_scores)

    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=kis_result,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.05,
    )
    by_id = _by_state_id(confs)

    assert by_id[1].flux == pytest.approx(10.0)
    assert by_id[2].flux == pytest.approx(20.0)
    assert by_id[1].kis_score == pytest.approx(2.0)
    assert by_id[2].kis_score == pytest.approx(3.0)


def test_macrostate_labels_attached_when_available_and_none_when_missing():
    """Macrostate labels attach when available and remain None when missing."""
    pi = np.array([0.2, 0.3, 0.5, 0.0])
    forward_committor = np.array([0.0, 0.25, 0.75, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(3,))
    flux_by_state = np.zeros_like(pi)

    macro_full = np.array([10, 11, 12, 13])
    confs_full = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=macro_full,
        tse_tolerance=0.05,
    )
    by_id_full = _by_state_id(confs_full)
    assert by_id_full[1].macrostate_id == 11
    assert by_id_full[2].macrostate_id == 12

    macro_short = np.array([10, 11])
    confs_short = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=macro_short,
        tse_tolerance=0.05,
    )
    by_id_short = _by_state_id(confs_short)
    assert by_id_short[1].macrostate_id == 11
    assert by_id_short[2].macrostate_id is None


def test_frame_index_is_placeholder_for_all_transition_conformations():
    """Transition and TSE conformations have placeholder frame indexes."""
    pi = np.array([0.2, 0.3, 0.5])
    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(2,))
    flux_by_state = np.zeros_like(pi)

    confs = m._find_transition_states(
        tpt_result=tpt,
        pi=pi,
        temperature_K=300.0,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        tse_tolerance=0.05,
    )

    assert len(confs) == 1
    assert confs[0].state_id == 1
    assert confs[0].frame_index == -1


def test_invalid_tse_tolerance_raises_value_error():
    """Invalid TSE tolerance ranges raise ValueError."""
    pi = np.array([0.2, 0.3, 0.5])
    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(2,))
    flux_by_state = np.zeros_like(pi)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=-0.01,
        )

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.51,
        )


def test_negative_temperature_raises_value_error():
    """Non-positive temperature raises ValueError."""
    pi = np.array([0.2, 0.3, 0.5])
    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(2,))
    flux_by_state = np.zeros_like(pi)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt,
            pi=pi,
            temperature_K=0.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt,
            pi=pi,
            temperature_K=-100.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )


def test_negative_population_raises_value_error():
    """Negative stationary probabilities raise ValueError."""
    pi = np.array([0.2, -0.1, 0.9])
    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(2,))
    flux_by_state = np.zeros_like(pi)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )


def test_length_mismatches_raise_value_error():
    """Length mismatches among required arrays raise ValueError."""
    pi = np.array([0.2, 0.3, 0.5])
    forward_committor_short = np.array([0.0, 0.5])
    tpt_bad_committor = DummyTPTResult(
        source_states=np.array([0]),
        sink_states=np.array([2]),
        forward_committor=forward_committor_short,
        flux_matrix=np.zeros((3, 3)),
    )
    flux_by_state = np.zeros_like(pi)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt_bad_committor,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )

    forward_committor = np.array([0.0, 0.5, 1.0])
    tpt_ok = _make_basic_tpt(pi, forward_committor, source_states=(0,), sink_states=(2,))
    bad_flux_by_state = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt_ok,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=bad_flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )

    kis_scores_bad = np.array([1.0, 2.0])
    kis_result_bad = DummyKISResult(kis_scores=kis_scores_bad)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt_ok,
            pi=pi,
            temperature_K=300.0,
            kis_result=kis_result_bad,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )


def test_source_and_sink_indices_out_of_range_raise_value_error():
    """Source and sink indices must lie within available state range."""
    pi = np.array([0.2, 0.3, 0.5])
    forward_committor = np.array([0.0, 0.5, 1.0])

    tpt_bad_source = DummyTPTResult(
        source_states=np.array([5]),
        sink_states=np.array([2]),
        forward_committor=forward_committor,
        flux_matrix=np.zeros((3, 3)),
    )
    flux_by_state = np.zeros_like(pi)

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt_bad_source,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )

    tpt_bad_sink = DummyTPTResult(
        source_states=np.array([0]),
        sink_states=np.array([7]),
        forward_committor=forward_committor,
        flux_matrix=np.zeros((3, 3)),
    )

    with pytest.raises(ValueError):
        m._find_transition_states(
            tpt_result=tpt_bad_sink,
            pi=pi,
            temperature_K=300.0,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            tse_tolerance=0.05,
        )
