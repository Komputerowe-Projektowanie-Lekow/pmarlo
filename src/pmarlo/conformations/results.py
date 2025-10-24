"""Result containers for conformations analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TPTResult:
    """Results from Transition Path Theory analysis.

    Attributes:
        source_states: Indices of source states
        sink_states: Indices of sink states
        forward_committor: Forward committor probabilities (q+)
        backward_committor: Backward committor probabilities (q-)
        flux_matrix: Reactive flux matrix between all states
        net_flux: Net reactive flux (forward - backward)
        total_flux: Total reactive flux from source to sink
        rate: Transition rate from source to sink (1/time)
        mfpt: Mean first passage time from source to sink
        pathways: Top reactive pathways (list of state sequences)
        pathway_fluxes: Flux contribution of each pathway
        bottleneck_states: States with highest reactive flux
    """

    source_states: np.ndarray
    sink_states: np.ndarray
    forward_committor: np.ndarray
    backward_committor: np.ndarray
    flux_matrix: np.ndarray
    net_flux: np.ndarray
    total_flux: float
    rate: float
    mfpt: float
    pathways: List[List[int]] = field(default_factory=list)
    pathway_fluxes: np.ndarray = field(default_factory=lambda: np.array([]))
    bottleneck_states: np.ndarray = field(default_factory=lambda: np.array([]))
    tpt_converged: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_states": self.source_states.tolist(),
            "sink_states": self.sink_states.tolist(),
            "forward_committor": self.forward_committor.tolist(),
            "backward_committor": self.backward_committor.tolist(),
            "flux_matrix": self.flux_matrix.tolist(),
            "net_flux": self.net_flux.tolist(),
            "total_flux": float(self.total_flux),
            "rate": float(self.rate),
            "mfpt": float(self.mfpt),
            "pathways": self.pathways,
            "pathway_fluxes": self.pathway_fluxes.tolist(),
            "bottleneck_states": self.bottleneck_states.tolist(),
            "tpt_converged": bool(self.tpt_converged),
        }


@dataclass(frozen=True, slots=True)
class KISResult:
    """Results from Kinetic Importance Score analysis.

    Attributes:
        kis_scores: KIS score for each microstate
        k_slow: Number of slow eigenvectors included
        eigenvectors: Slow eigenvectors used (k_slow x n_states)
        eigenvalues: Corresponding eigenvalues
        ranked_states: States ranked by KIS (descending)
        stability_metric: Measure of KIS ranking stability (if computed)
        bootstrap_std: Bootstrap standard deviation of KIS (if computed)
    """

    kis_scores: np.ndarray
    k_slow: int
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    ranked_states: np.ndarray
    stability_metric: Optional[float] = None
    bootstrap_std: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "kis_scores": self.kis_scores.tolist(),
            "k_slow": int(self.k_slow),
            "eigenvectors": self.eigenvectors.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "ranked_states": self.ranked_states.tolist(),
            "stability_metric": (
                float(self.stability_metric) if self.stability_metric is not None else None
            ),
            "bootstrap_std": (
                self.bootstrap_std.tolist() if self.bootstrap_std is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class UncertaintyResult:
    """Results from uncertainty quantification analysis.

    Attributes:
        observable_name: Name of the observable
        mean: Mean value across bootstrap/ensemble
        std: Standard deviation
        ci_lower: Lower confidence interval (default 2.5%)
        ci_upper: Upper confidence interval (default 97.5%)
        n_samples: Number of bootstrap samples or ensemble models
        method: 'bootstrap' or 'hyperparameter_ensemble'
    """

    observable_name: str
    mean: float | np.ndarray
    std: float | np.ndarray
    ci_lower: float | np.ndarray
    ci_upper: float | np.ndarray
    n_samples: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

        def _to_serializable(val: Any) -> Any:
            if isinstance(val, np.ndarray):
                return val.tolist()
            return float(val) if np.isscalar(val) else val

        return {
            "observable_name": self.observable_name,
            "mean": _to_serializable(self.mean),
            "std": _to_serializable(self.std),
            "ci_lower": _to_serializable(self.ci_lower),
            "ci_upper": _to_serializable(self.ci_upper),
            "n_samples": int(self.n_samples),
            "method": str(self.method),
        }


@dataclass(slots=True)
class Conformation:
    """Single conformation with metadata.

    Attributes:
        conformation_type: Type of conformation ('metastable', 'transition', 'tse', 'pathway')
        state_id: Microstate ID in the MSM
        macrostate_id: Macrostate ID (if applicable)
        frame_index: Global frame index in concatenated trajectories
        trajectory_index: Trajectory file index
        local_frame_index: Frame index within trajectory
        population: Stationary population of this state
        free_energy: Free energy (kJ/mol)
        committor: Committor probability (available for all states when TPT is computed)
        kis_score: Kinetic importance score (if computed)
        flux: Reactive flux through the state
        structure_path: Path to saved PDB file (if extracted)
        metadata: Additional metadata
    """

    conformation_type: str
    state_id: int
    frame_index: int
    population: float
    free_energy: float
    macrostate_id: Optional[int] = None
    trajectory_index: Optional[int] = None
    local_frame_index: Optional[int] = None
    committor: Optional[float] = None
    kis_score: Optional[float] = None
    flux: Optional[float] = None
    structure_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conformation_type": self.conformation_type,
            "state_id": int(self.state_id),
            "macrostate_id": int(self.macrostate_id) if self.macrostate_id is not None else None,
            "frame_index": int(self.frame_index),
            "trajectory_index": int(self.trajectory_index) if self.trajectory_index is not None else None,
            "local_frame_index": int(self.local_frame_index) if self.local_frame_index is not None else None,
            "population": float(self.population),
            "free_energy": float(self.free_energy),
            "committor": float(self.committor) if self.committor is not None else None,
            "kis_score": float(self.kis_score) if self.kis_score is not None else None,
            "flux": float(self.flux) if self.flux is not None else None,
            "structure_path": str(self.structure_path) if self.structure_path else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ConformationSet:
    """Complete set of conformations from analysis.

    Attributes:
        conformations: List of all conformations
        tpt_result: TPT analysis results (if computed)
        kis_result: KIS analysis results (if computed)
        uncertainty_results: Uncertainty quantification results
        macrostate_labels: Macrostate assignment for each microstate
        metadata: Analysis metadata (parameters, timestamps, etc.)
    """

    conformations: List[Conformation] = field(default_factory=list)
    tpt_result: Optional[TPTResult] = None
    kis_result: Optional[KISResult] = None
    uncertainty_results: List[UncertaintyResult] = field(default_factory=list)
    macrostate_labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_by_type(self, conformation_type: str) -> List[Conformation]:
        """Get all conformations of a specific type.

        Args:
            conformation_type: 'metastable', 'transition', or 'pathway'

        Returns:
            List of matching conformations
        """
        return [c for c in self.conformations if c.conformation_type == conformation_type]

    def get_transition_states(self) -> List[Conformation]:
        """Get all transition state conformations."""
        return self.get_by_type("transition")

    def get_transition_state_ensemble(self) -> List[Conformation]:
        """Get transition state ensemble conformations (committor ≈ 0.5)."""
        return self.get_by_type("tse")

    def get_metastable_states(self) -> List[Conformation]:
        """Get all metastable state conformations."""
        return self.get_by_type("metastable")

    def get_pathway_intermediates(self) -> List[Conformation]:
        """Backward-compatible alias for reactive transition states."""
        # Classification now treats all non-source/sink states as transition states.
        return self.get_transition_states()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conformations": [c.to_dict() for c in self.conformations],
            "tpt_result": self.tpt_result.to_dict() if self.tpt_result else None,
            "kis_result": self.kis_result.to_dict() if self.kis_result else None,
            "uncertainty_results": [u.to_dict() for u in self.uncertainty_results],
            "macrostate_labels": (
                self.macrostate_labels.tolist() if self.macrostate_labels is not None else None
            ),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Save to JSON file.

        Args:
            filepath: Path to output file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, filepath: str | Path) -> ConformationSet:
        """Load from JSON file.

        Args:
            filepath: Path to input file

        Returns:
            ConformationSet instance
        """
        data = json.loads(Path(filepath).read_text())

        # Reconstruct conformations
        conformations = [Conformation(**c) for c in data.get("conformations", [])]

        # Reconstruct TPT result
        tpt_data = data.get("tpt_result")
        tpt_result = None
        if tpt_data:
            tpt_result = TPTResult(
                source_states=np.array(tpt_data["source_states"]),
                sink_states=np.array(tpt_data["sink_states"]),
                forward_committor=np.array(tpt_data["forward_committor"]),
                backward_committor=np.array(tpt_data["backward_committor"]),
                flux_matrix=np.array(tpt_data["flux_matrix"]),
                net_flux=np.array(tpt_data["net_flux"]),
                total_flux=tpt_data["total_flux"],
                rate=tpt_data["rate"],
                mfpt=tpt_data["mfpt"],
                pathways=tpt_data.get("pathways", []),
                pathway_fluxes=np.array(tpt_data.get("pathway_fluxes", [])),
                bottleneck_states=np.array(tpt_data.get("bottleneck_states", [])),
                tpt_converged=bool(tpt_data.get("tpt_converged", True)),
            )

        # Reconstruct KIS result
        kis_data = data.get("kis_result")
        kis_result = None
        if kis_data:
            kis_result = KISResult(
                kis_scores=np.array(kis_data["kis_scores"]),
                k_slow=kis_data["k_slow"],
                eigenvectors=np.array(kis_data["eigenvectors"]),
                eigenvalues=np.array(kis_data["eigenvalues"]),
                ranked_states=np.array(kis_data["ranked_states"]),
                stability_metric=kis_data.get("stability_metric"),
                bootstrap_std=(
                    np.array(kis_data["bootstrap_std"])
                    if kis_data.get("bootstrap_std")
                    else None
                ),
            )

        # Reconstruct uncertainty results
        unc_data = data.get("uncertainty_results", [])
        uncertainty_results = []
        for u in unc_data:
            uncertainty_results.append(
                UncertaintyResult(
                    observable_name=u["observable_name"],
                    mean=np.array(u["mean"]) if isinstance(u["mean"], list) else u["mean"],
                    std=np.array(u["std"]) if isinstance(u["std"], list) else u["std"],
                    ci_lower=(
                        np.array(u["ci_lower"])
                        if isinstance(u["ci_lower"], list)
                        else u["ci_lower"]
                    ),
                    ci_upper=(
                        np.array(u["ci_upper"])
                        if isinstance(u["ci_upper"], list)
                        else u["ci_upper"]
                    ),
                    n_samples=u["n_samples"],
                    method=u["method"],
                )
            )

        macrostate_labels = None
        if data.get("macrostate_labels") is not None:
            macrostate_labels = np.array(data["macrostate_labels"])

        return cls(
            conformations=conformations,
            tpt_result=tpt_result,
            kis_result=kis_result,
            uncertainty_results=uncertainty_results,
            macrostate_labels=macrostate_labels,
            metadata=data.get("metadata", {}),
        )

