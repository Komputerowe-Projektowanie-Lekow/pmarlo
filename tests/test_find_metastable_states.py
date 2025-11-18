"""Behavioral tests for `_find_metastable_states`."""

from dataclasses import dataclass

import numpy as np
import pytest

from pmarlo.conformations.finder import _find_metastable_states


@dataclass
class DummyTPTResult:
    """Minimal object with the attributes _find_metastable_states needs."""

    source_states: np.ndarray
    sink_states: np.ndarray
    forward_committor: np.ndarray
    flux_matrix: np.ndarray


@dataclass
class DummyKISResult:
    """Minimal object with the attributes _find_metastable_states needs."""

    kis_scores: np.ndarray


def test_metastable_states_basic_with_all_metadata():
    """
    Union of source and sink states is used.
    Per-state quantities (population, free energy, committor, flux, KIS,
    macrostate info, role) are mapped correctly.
    """
    pi = np.array([0.5, 0.2, 0.2, 0.1])
    T = 300.0

    source_states = np.array([0, 1, 2])
    sink_states = np.array([2, 3])  # 2 is both source and sink

    forward_committor = np.array([0.0, 0.3, 0.7, 1.0])
    flux_by_state = np.array([1.0, 2.0, 3.0, 4.0])
    kis_scores = np.array([0.1, 0.2, 0.3, 0.4])

    macrostate_labels = np.array([10, 11, 12, 13])
    macrostate_memberships = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ]
    )

    tpt_result = DummyTPTResult(
        source_states=source_states,
        sink_states=sink_states,
        forward_committor=forward_committor,
        flux_matrix=np.zeros((4, 4)),  # will not be used because we pass flux_by_state
    )
    kis_result = DummyKISResult(kis_scores=kis_scores)

    labels_out, conformations = _find_metastable_states(
        tpt_result=tpt_result,
        pi=pi,
        temperature_K=T,
        kis_result=kis_result,
        flux_by_state=flux_by_state,
        macrostate_labels=macrostate_labels,
        macrostate_memberships=macrostate_memberships,
    )

    # macrostate_labels should be passed through (or equivalent)
    assert np.array_equal(labels_out, macrostate_labels)

    # Metastable state IDs must be exactly source ∪ sink
    returned_state_ids = {c.state_id for c in conformations}
    expected_state_ids = set(source_states).union(set(sink_states))
    assert returned_state_ids == expected_state_ids

    # Convenience mapping
    conf_by_state = {c.state_id: c for c in conformations}

    # Roles:
    #  - purely source -> "source"
    #  - purely sink -> "sink"
    #  - in both -> "source_sink"
    assert conf_by_state[0].metadata["role"] == "source"
    assert conf_by_state[1].metadata["role"] == "source"
    assert conf_by_state[2].metadata["role"] == "source_sink"
    assert conf_by_state[3].metadata["role"] == "sink"

    # Check one representative state in detail (state 1)
    s = 1
    conf = conf_by_state[s]

    assert conf.conformation_type == "metastable"
    assert conf.frame_index == -1  # no representative frame yet

    # Per-state quantities
    assert conf.population == pytest.approx(pi[s])
    assert conf.committor == pytest.approx(forward_committor[s])
    assert conf.flux == pytest.approx(flux_by_state[s])
    assert conf.kis_score == pytest.approx(kis_scores[s])
    assert conf.macrostate_id == macrostate_labels[s]
    assert conf.metadata["macrostate_members"] == pytest.approx(
        macrostate_memberships[s].tolist()
    )

    # Free energy should match -kT ln(pi)
    # Use R = 8.314462618... J/mol/K, convert to kJ/mol
    R = 8.31446261815324
    kT = R * T / 1000.0
    expected_fe = -kT * np.log(pi[s])
    assert conf.free_energy == pytest.approx(expected_fe)


def test_no_kis_no_macrostates():
    """
    When KIS and macrostate info are not provided, those fields should be None
    or absent, but metastable selection still works.
    """
    pi = np.array([0.6, 0.4])
    T = 310.0
    source_states = np.array([0])
    sink_states = np.array([1])

    forward_committor = np.array([0.0, 1.0])
    flux_by_state = np.array([1.5, 2.5])

    tpt_result = DummyTPTResult(
        source_states=source_states,
        sink_states=sink_states,
        forward_committor=forward_committor,
        flux_matrix=np.zeros((2, 2)),
    )

    labels_out, conformations = _find_metastable_states(
        tpt_result=tpt_result,
        pi=pi,
        temperature_K=T,
        kis_result=None,
        flux_by_state=flux_by_state,
        macrostate_labels=None,
        macrostate_memberships=None,
    )

    assert labels_out is None
    assert {c.state_id for c in conformations} == {0, 1}

    conf_by_state = {c.state_id: c for c in conformations}

    # KIS and macrostate-related fields must be unset
    assert conf_by_state[0].kis_score is None
    assert conf_by_state[1].kis_score is None
    assert conf_by_state[0].macrostate_id is None
    assert conf_by_state[1].macrostate_id is None
    assert "macrostate_members" not in conf_by_state[0].metadata
    assert "macrostate_members" not in conf_by_state[1].metadata


def test_empty_source_and_sink_returns_empty_list():
    """
    If both source_states and sink_states are empty, there are no metastable
    states and the function should return an empty list without crashing.
    """
    pi = np.array([0.5, 0.5])
    T = 300.0

    tpt_result = DummyTPTResult(
        source_states=np.array([], dtype=int),
        sink_states=np.array([], dtype=int),
        forward_committor=np.array([0.0, 1.0]),
        flux_matrix=np.zeros((2, 2)),
    )

    labels_out, conformations = _find_metastable_states(
        tpt_result=tpt_result,
        pi=pi,
        temperature_K=T,
        kis_result=None,
        flux_by_state=None,  # should not be used in this case
        macrostate_labels=None,
        macrostate_memberships=None,
    )

    assert labels_out is None
    assert conformations == []


def test_temperature_must_be_positive():
    """
    Non-positive temperature is physically meaningless and should raise.
    """
    pi = np.array([0.5, 0.5])
    T = 0.0

    tpt_result = DummyTPTResult(
        source_states=np.array([0]),
        sink_states=np.array([1]),
        forward_committor=np.array([0.0, 1.0]),
        flux_matrix=np.zeros((2, 2)),
    )

    with pytest.raises(ValueError):
        _find_metastable_states(
            tpt_result=tpt_result,
            pi=pi,
            temperature_K=T,
            kis_result=None,
            flux_by_state=np.array([1.0, 2.0]),
            macrostate_labels=None,
            macrostate_memberships=None,
        )


def test_state_indices_out_of_range_raise():
    """
    If metastable state indices exceed the length of pi, the function should
    raise instead of silently producing nonsense.
    """
    pi = np.array([0.5, 0.5])  # states 0 and 1 only
    T = 300.0

    # State 2 does not exist in pi
    source_states = np.array([0, 2])
    sink_states = np.array([], dtype=int)

    tpt_result = DummyTPTResult(
        source_states=source_states,
        sink_states=sink_states,
        forward_committor=np.array([0.0, 1.0]),
        flux_matrix=np.zeros((2, 2)),
    )

    with pytest.raises(ValueError):
        _find_metastable_states(
            tpt_result=tpt_result,
            pi=pi,
            temperature_K=T,
            kis_result=None,
            flux_by_state=np.array([1.0, 2.0]),
            macrostate_labels=None,
            macrostate_memberships=None,
        )


def test_flux_by_state_too_short_raises():
    """
    If flux_by_state does not cover all metastable states, this should raise.
    """
    pi = np.array([0.4, 0.3, 0.3])
    T = 300.0

    source_states = np.array([0, 1, 2])
    sink_states = np.array([], dtype=int)

    tpt_result = DummyTPTResult(
        source_states=source_states,
        sink_states=sink_states,
        forward_committor=np.array([0.0, 0.5, 1.0]),
        flux_matrix=np.zeros((3, 3)),
    )

    # flux_by_state has length 2 but we need index 2 as well
    flux_by_state = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        _find_metastable_states(
            tpt_result=tpt_result,
            pi=pi,
            temperature_K=T,
            kis_result=None,
            flux_by_state=flux_by_state,
            macrostate_labels=None,
            macrostate_memberships=None,
        )
