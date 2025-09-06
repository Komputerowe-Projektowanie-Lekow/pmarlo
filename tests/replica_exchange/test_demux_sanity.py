from __future__ import annotations

import inspect
from pathlib import Path

import typing as t

from pmarlo.replica_exchange.demux import demux_trajectories
from pmarlo.replica_exchange.demux_metadata import DemuxIntegrityError, DemuxMetadata
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def test_demux_function_signature_stable() -> None:
    """Public API: function and method signatures remain stable.

    Guardrails to avoid accidental breaking changes during refactors.
    """

    # Module-level function signature
    f_sig = inspect.signature(demux_trajectories)
    f_params = list(f_sig.parameters.values())
    assert f_params[0].name == "remd"
    assert f_params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    # Keyword-only parameters after the * sentinel
    expected_kwonly = ["target_temperature", "equilibration_steps", "progress_callback"]
    actual_kwonly = [p.name for p in f_params[1:] if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert actual_kwonly == expected_kwonly

    # Return type: Optional[str]
    from typing import get_args, get_origin

    ra = f_sig.return_annotation
    if isinstance(ra, str):
        # Deferred annotations (from __future__ import annotations); accept string form
        assert "Optional" in ra and "str" in ra
    else:
        origin = get_origin(ra)
        args = get_args(ra)
        assert origin in {t.Union, t.Optional}
        assert set(args) == {str, type(None)}

    # Bound method signature on the class
    m_sig = inspect.signature(ReplicaExchange.demux_trajectories)
    m_params = list(m_sig.parameters.values())
    assert m_params[0].name == "self"
    assert m_params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    expected_kwonly_m = ["target_temperature", "equilibration_steps", "progress_callback"]
    actual_kwonly_m = [p.name for p in m_params[1:] if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert actual_kwonly_m == expected_kwonly_m


def test_missing_trajectory_file_is_handled_gracefully(tmp_path: Path) -> None:
    """Missing files do not surface raw I/O errors during demux.

    The current behavior logs and returns None if no trajectories are loadable.
    """

    # Construct a minimal REMD-like object without running OpenMM
    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = str(tmp_path / "nonexistent.pdb")  # unused when files are missing
    remd.temperatures = [300.0]
    remd.n_replicas = 1
    remd.exchange_history = [[0], [0]]  # ensure demux path is exercised
    remd.reporter_stride = None
    remd.dcd_stride = 1
    remd.exchange_frequency = 1
    remd.output_dir = tmp_path
    remd.integrators = []
    remd._replica_reporter_stride = []
    remd.trajectory_files = [tmp_path / "replica_00.dcd"]  # does not exist

    # Should not raise FileNotFoundError or other raw I/O; returns None
    path = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)
    assert path is None


def test_exposed_project_specific_exceptions_present() -> None:
    """Sanity: project-specific demux exception and metadata types are importable."""

    assert DemuxIntegrityError is not None
    # Minimal check that metadata type exposes required fields
    fields = set(getattr(DemuxMetadata, "__dataclass_fields__").keys())  # type: ignore[attr-defined]
    expected = {
        "exchange_frequency_steps",
        "integration_timestep_ps",
        "frames_per_segment",
        "temperature_schedule",
    }
    assert expected.issubset(fields)
