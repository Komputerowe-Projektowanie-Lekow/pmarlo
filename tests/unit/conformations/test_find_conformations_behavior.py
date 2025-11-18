"""Black-box behavioral tests for find_conformations."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.finder import find_conformations


def _two_state_msm() -> tuple[np.ndarray, np.ndarray]:
    """Simple reversible 2-state MSM for conceptual tests."""
    T = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=float,
    )
    pi = np.array([0.5, 0.5], dtype=float)
    return T, pi


# -------------------------------------------------------------------
# 1. Input validation behaviour
# -------------------------------------------------------------------


def test_requires_T_and_pi() -> None:
    """find_conformations must reject msm_data without T or pi."""
    T, pi = _two_state_msm()

    with pytest.raises(ValueError):
        find_conformations(msm_data={})

    with pytest.raises(ValueError):
        find_conformations(msm_data={"T": T})

    with pytest.raises(ValueError):
        find_conformations(msm_data={"pi": pi})


def test_uncertainty_requires_dtrajs() -> None:
    """uncertainty_analysis=True requires 'dtrajs' in msm_data."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    with pytest.raises(ValueError):
        find_conformations(
            msm_data,
            source_states=np.array([0]),
            sink_states=np.array([1]),
            auto_detect=False,
            uncertainty_analysis=True,
        )


def test_save_structures_requires_output_dir() -> None:
    """save_structures=True mandates an output directory."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    with pytest.raises(ValueError):
        find_conformations(
            msm_data,
            source_states=np.array([0]),
            sink_states=np.array([1]),
            auto_detect=False,
            save_structures=True,
            output_dir=None,
        )


# -------------------------------------------------------------------
# 2. Source / sink semantics and auto-detection
# -------------------------------------------------------------------


def test_manual_source_sink_overrides_auto_detect() -> None:
    """Explicit inputs must win over auto detection."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    result = find_conformations(
        msm_data,
        source_states=np.array([0]),
        sink_states=np.array([1]),
        auto_detect=True,
        find_metastable_states=False,
        find_transition_states=False,
        find_pathway_intermediates=False,
        compute_kis=False,
        uncertainty_analysis=False,
        save_structures=False,
    )

    assert hasattr(result, "metadata")
    assert result.metadata.get("auto_detected") is False


def test_auto_detect_used_when_no_manual_source_sink() -> None:
    """Auto detection should engage (and be recorded) when inputs missing."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    result = find_conformations(
        msm_data,
        source_states=None,
        sink_states=None,
        auto_detect=True,
        find_metastable_states=False,
        find_transition_states=False,
        find_pathway_intermediates=False,
        compute_kis=False,
        uncertainty_analysis=False,
        save_structures=False,
    )

    assert hasattr(result, "metadata")
    assert result.metadata.get("auto_detected") is True


# -------------------------------------------------------------------
# 3. Flag behaviour: KIS and conformational classification
# -------------------------------------------------------------------


def test_compute_kis_flag_disables_kis_when_false() -> None:
    """compute_kis=False should omit KIS information."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    result = find_conformations(
        msm_data,
        source_states=np.array([0]),
        sink_states=np.array([1]),
        auto_detect=False,
        find_metastable_states=False,
        find_transition_states=False,
        find_pathway_intermediates=False,
        compute_kis=False,
        uncertainty_analysis=False,
        save_structures=False,
    )

    assert hasattr(result, "kis_result")
    assert result.kis_result is None


def test_two_state_msm_yields_two_metastable_conformations() -> None:
    """Integration-style test on a minimal MSM."""
    T, pi = _two_state_msm()
    msm_data = {"T": T, "pi": pi}

    result = find_conformations(
        msm_data,
        source_states=np.array([0]),
        sink_states=np.array([1]),
        auto_detect=False,
        find_metastable_states=True,
        find_transition_states=False,
        find_pathway_intermediates=False,
        compute_kis=False,
        uncertainty_analysis=False,
        save_structures=False,
    )

    assert hasattr(result, "conformations")
    conformations = result.conformations
    assert len(conformations) == 2
    assert all(c.conformation_type == "metastable" for c in conformations)

    state_ids = {c.state_id for c in conformations}
    assert state_ids == {0, 1}

    assert hasattr(result, "metadata")
    assert result.metadata.get("n_conformations") == 2
    assert result.metadata.get("n_metastable_states") == 2
