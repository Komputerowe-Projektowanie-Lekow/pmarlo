"""Lightweight facade for replica-exchange simulation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    try:
        from .bias_hook import BiasHook
    except ImportError:
        BiasHook = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

_FULL_IMPORT_ERROR: Exception | None = None


def _load_full_impl() -> bool:
    """Attempt to import the heavy-weight simulation backend."""
    import importlib.util

    global _FULL_IMPORT_ERROR

    if importlib.util.find_spec("openmm") is None:
        _FULL_IMPORT_ERROR = ImportError(
            "OpenMM not available; replica-exchange simulation requires optional dependencies."
        )
        return False

    try:  # pragma: no cover - exercised only when OpenMM is present
        from ._simulation_full import (
            Simulation as _FullSimulation,  # type: ignore[assignment]
        )
        from ._simulation_full import (
            build_transition_model,
            plot_DG,
            prepare_system,
            production_run,
            relative_energies,
        )
    except Exception as exc:  # pragma: no cover - optional dependency missing
        _FULL_IMPORT_ERROR = exc
        return False

    globals().update(
        Simulation=_FullSimulation,
        build_transition_model=build_transition_model,
        plot_DG=plot_DG,
        prepare_system=prepare_system,
        production_run=production_run,
        relative_energies=relative_energies,
    )
    return True


_HAS_FULL_IMPL = _load_full_impl()


if not _HAS_FULL_IMPL:

    @dataclass
    class Simulation:  # type: ignore[no-redef]
        """Minimal placeholder implementation used when OpenMM is unavailable."""

        pdb_file: str
        output_dir: str = "output"
        temperature: float = 300.0
        steps: int = 1000
        use_metadynamics: bool = True
        platform: str = "CPU"
        random_seed: int | None = None

        def __post_init__(self) -> None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            logger.warning(
                "OpenMM stack not available; Simulation acts as a lightweight stub."
            )

        def prepare_system(self, *args: Any, **kwargs: Any) -> tuple[None, None]:
            raise ImportError(
                "Simulation.prepare_system requires OpenMM."
                " Install with `pip install 'pmarlo[full]'`."
            ) from _FULL_IMPORT_ERROR

        def run_production(self, *args: Any, **kwargs: Any) -> str:
            raise ImportError(
                "Simulation.run_production requires OpenMM."
                " Install with `pip install 'pmarlo[full]'`."
            ) from _FULL_IMPORT_ERROR

        def feature_extraction(
            self, *_args: Any, **_kwargs: Any
        ) -> Dict[str, np.ndarray]:
            raise ImportError(
                "Simulation.feature_extraction requires OpenMM+mdtraj."
                " Install with `pip install 'pmarlo[full]'`."
            ) from _FULL_IMPORT_ERROR

    def prepare_system(
        pdb_file: Any, forcefield_files: Any = None, water_model: Any = "tip3p"
    ) -> Any:  # type: ignore[misc]
        raise ImportError(
            "prepare_system requires OpenMM."
            " Install with `pip install 'pmarlo[full]'`."
        ) from _FULL_IMPORT_ERROR

    def production_run(
        sim: Any,
        steps: Any = 100000,
        report_interval: Any = 1000,
        bias_hook: "BiasHook | None" = None,
    ) -> Any:  # type: ignore[misc]
        raise ImportError(
            "production_run requires OpenMM."
            " Install with `pip install 'pmarlo[full]'`."
        ) from _FULL_IMPORT_ERROR

    def build_transition_model(
        features: Any, n_states: Any = 50, lag_time: Any = 1
    ) -> Any:  # type: ignore[misc]
        raise ImportError(
            "build_transition_model requires the analysis stack (scikit-learn)."
            " Install with `pip install 'pmarlo[analysis]'`."
        ) from _FULL_IMPORT_ERROR

    def relative_energies(
        msm_result: Any, reference_state: Any = 0
    ) -> Any:  # type: ignore[misc]
        raise ImportError(
            "relative_energies requires the analysis stack."
            " Install with `pip install 'pmarlo[analysis]'`."
        ) from _FULL_IMPORT_ERROR

    def plot_DG(features: Any, save_path: Any = None) -> Any:  # type: ignore[misc]
        raise ImportError(
            "plot_DG requires matplotlib. Install with `pip install 'pmarlo[plot]'`."
        ) from _FULL_IMPORT_ERROR


def feature_extraction(
    trajectory_file: str,
    topology_file: str,
    *,
    random_state: int | None = None,
    n_states: int = 40,
    stride: int = 1,
    **cluster_kwargs: Any,
) -> np.ndarray:
    """Cluster trajectory frames into microstates using lightweight defaults.

    Parameters
    ----------
    trajectory_file:
        Path to the trajectory file (DCD).  Only Cartesian coordinates are used.
    topology_file:
        Matching topology file (PDB) describing the atoms in the trajectory.
    random_state:
        Seed forwarded to :func:`pmarlo.api.cluster_microstates` for deterministic
        clustering.  ``None`` keeps the backend default.
    n_states:
        Target number of microstates.  Defaults to 40 for backwards compatibility
        with earlier workflows.
    stride:
        Optional frame thinning factor when loading the trajectory.
    **cluster_kwargs:
        Additional keyword arguments forwarded verbatim to the clustering API.
    """

    try:
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ImportError(
            "feature_extraction requires mdtraj. Install with `pip install 'pmarlo[full]'`."
        ) from exc

    try:
        from pmarlo import api
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ImportError(
            "pmarlo.api is unavailable; install scikit-learn with `pmarlo[analysis]`"
        ) from exc

    stride_int = max(1, int(stride))
    logger.info(
        "Loading trajectory '%s' with topology '%s' (stride=%d)",
        trajectory_file,
        topology_file,
        stride_int,
    )
    traj = md.load(trajectory_file, top=topology_file, stride=stride_int)
    if traj.n_frames == 0:
        raise ValueError(
            "Loaded trajectory contains no frames; cannot extract features"
        )

    coords = traj.xyz.reshape(traj.n_frames, -1)
    cluster_args: Dict[str, Any] = {
        "method": cluster_kwargs.pop("method", "auto"),
        "n_states": cluster_kwargs.pop("n_states", n_states),
        "random_state": random_state,
    }
    cluster_args.update(cluster_kwargs)

    logger.info(
        "Clustering %d frames into %s states (method=%s)",
        coords.shape[0],
        cluster_args.get("n_states", n_states),
        cluster_args.get("method", "auto"),
    )
    labels = api.cluster_microstates(coords, **cluster_args)
    return np.asarray(labels)
