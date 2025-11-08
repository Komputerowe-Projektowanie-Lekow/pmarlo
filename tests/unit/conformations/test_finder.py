"""Unit tests for the high-level conformations finder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pmarlo.conformations import find_conformations
from pmarlo.conformations.finder import _extract_structures
from pmarlo.conformations.results import Conformation


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a transition matrix."""

    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = int(np.argmax(np.real(eigenvalues)))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi = pi / np.sum(pi)
    return pi.astype(float)


def test_find_conformations_uses_pcca_macrostates() -> None:
    """PCCA+ macrostates are exposed as metastable conformations."""

    T = np.array(
        [
            [0.92, 0.08, 0.0, 0.0],
            [0.08, 0.9, 0.02, 0.0],
            [0.0, 0.05, 0.9, 0.05],
            [0.0, 0.0, 0.08, 0.92],
        ],
        dtype=float,
    )
    T = T / T.sum(axis=1, keepdims=True)
    pi = _stationary_distribution(T)

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [np.tile(np.arange(T.shape[0]), 10)],
    }

    results = find_conformations(
        msm_data=msm_data,
        auto_detect=True,
        find_metastable_states=True,
        find_transition_states=False,
        compute_kis=False,
        n_metastable=2,
    )

    assert results.macrostate_labels is not None
    assert set(np.unique(results.macrostate_labels)) == {0, 1}
    memberships = np.asarray(results.metadata["macrostate_memberships"])
    assert memberships.shape == (T.shape[0], 2)
    np.testing.assert_allclose(memberships.sum(axis=1), 1.0, atol=1e-8)
    assert results.metadata["n_metastable_states"] == 2

    metastable = results.get_metastable_states()
    assert len(metastable) == 2

    roles = {conf.metadata.get("role") for conf in metastable}
    assert roles == {"source", "sink"}

    for conf in metastable:
        assert conf.macrostate_id in (0, 1)
        members = np.asarray(conf.metadata["macrostate_members"])
        assert members.shape == (2,)
        np.testing.assert_allclose(np.sum(members), 1.0, atol=1e-8)


def test_find_conformations_identifies_tse_states() -> None:
    """Transition state ensemble microstates are separated from intermediates."""

    T = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )
    T = T / T.sum(axis=1, keepdims=True)
    pi = _stationary_distribution(T)

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [np.tile(np.arange(T.shape[0]), 20)],
    }

    results = find_conformations(
        msm_data=msm_data,
        auto_detect=False,
        source_states=np.array([0]),
        sink_states=np.array([2]),
        find_metastable_states=False,
        find_transition_states=True,
        compute_kis=False,
        tse_tolerance=0.2,
        n_metastable=3,
    )

    tse_conformations = results.get_transition_state_ensemble()
    assert len(tse_conformations) >= 1

    for conf in tse_conformations:
        assert conf.conformation_type == "tse"
        assert abs(conf.committor - 0.5) <= 0.2 + 1e-6

    assert results.metadata["n_transition_state_ensemble"] == len(tse_conformations)

    other_transitions = [
        c for c in results.conformations if c.conformation_type == "transition"
    ]
    for conf in other_transitions:
        assert conf.conformation_type == "transition"


def test_extract_structures_forwards_locator_and_topology(monkeypatch: Any, tmp_path: Path) -> None:
    """Structure extraction uses locator/topology arguments."""

    topology_path = tmp_path / "topology.pdb"
    topology_path.write_text("HEADER")
    trajectory_locator = object()
    called: dict[str, Any] = {}

    def fake_extract(
        self,  # type: ignore[unused-arg]
        representatives,
        trajectories,
        output_dir,
        prefix,
        *,
        topology_path=None,
        trajectory_locator=None,
    ):
        called["representatives"] = representatives
        called["trajectories"] = trajectories
        called["output_dir"] = output_dir
        called["prefix"] = prefix
        called["topology_path"] = topology_path
        called["trajectory_locator"] = trajectory_locator
        return []

    monkeypatch.setattr(
        "pmarlo.conformations.finder.RepresentativePicker.extract_structures",
        fake_extract,
    )

    conf = Conformation(
        conformation_type="metastable",
        state_id=0,
        frame_index=0,
        population=0.5,
        free_energy=0.0,
        trajectory_index=0,
        local_frame_index=0,
    )

    _extract_structures(
        [conf],
        trajectories=None,
        output_dir=str(tmp_path / "structures"),
        topology_path=str(topology_path),
        trajectory_locator=trajectory_locator,
    )

    assert called["topology_path"] == str(topology_path)
    assert called["trajectory_locator"] is trajectory_locator
