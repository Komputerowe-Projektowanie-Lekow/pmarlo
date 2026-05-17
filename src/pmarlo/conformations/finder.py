"""High-level API for finding conformations using TPT analysis."""

from __future__ import annotations

import logging
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pmarlo.utils.thermodynamics import kT_kJ_per_mol

from .kinetic_importance import KineticImportanceScore
from .representative_picker import (
    RepresentativePicker,
    TrajectoryFrameLocator,
)
from .results import (
    Conformation,
    ConformationSet,
    KISResult,
    TPTResult,
    UncertaintyResult,
)
from .state_detection import StateDetector
from .tpt_analysis import TPTAnalysis
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger("pmarlo.conformations")


def _resolve_n_metastable(requested: Optional[int], n_states: int) -> int:
    """Resolve the requested number of macrostates with sanity checks."""
    resolved = 2 if requested is None else int(requested)
    if resolved < 2:
        raise ValueError("n_metastable must be at least 2")
    if resolved > n_states:
        raise ValueError(
            f"n_metastable ({resolved}) cannot exceed the number of MSM states ({n_states})"
        )
    return resolved


def _compute_macrostate_memberships(
    T: np.ndarray,
    pi: np.ndarray,
    n_metastable: int,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-7,
    stationary_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCCA+ memberships and canonical macrostate labels.

    The returned memberships are ordered by decreasing stationary macrostate
    population, so macrostate label 0 corresponds to the most populated
    macrostate.

    Parameters
    ----------
    T:
        Row-stochastic transition matrix of shape ``(n_microstates, n_microstates)``.
    pi:
        Stationary distribution over microstates of shape ``(n_microstates,)``.
    n_metastable:
        Number of metastable macrostates requested.
    atol:
        Absolute tolerance for probability and row-stochasticity checks.
    rtol:
        Relative tolerance for probability and row-stochasticity checks.
    stationary_atol:
        Absolute tolerance for checking ``pi @ T == pi``.

    Returns
    -------
    memberships:
        Array of shape ``(n_microstates, n_metastable)``. Each row contains
        fuzzy membership probabilities over macrostates.
    macrostate_labels:
        Hard macrostate labels obtained by ``argmax`` over canonicalized
        memberships.
    """
    T_arr = np.asarray(T, dtype=float)

    if T_arr.ndim != 2 or T_arr.shape[0] != T_arr.shape[1]:
        raise ValueError("Transition matrix must be square to compute macrostates.")

    n_states = T_arr.shape[0]

    if not isinstance(n_metastable, Integral):
        raise TypeError("n_metastable must be an integer.")

    n_metastable = int(n_metastable)

    if n_metastable < 1:
        raise ValueError("n_metastable must be at least 1.")

    if n_metastable > n_states:
        raise ValueError(
            f"Requested {n_metastable} macrostates, but only {n_states} "
            "microstates are available."
        )

    if not np.all(np.isfinite(T_arr)):
        raise ValueError("Transition matrix contains NaN or infinite values.")

    if np.any(T_arr < -atol):
        min_value = float(np.min(T_arr))
        raise ValueError(
            f"Transition matrix contains negative probabilities. "
            f"Minimum value: {min_value:.3e}."
        )

    row_sums = T_arr.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=atol, rtol=rtol):
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            "Transition matrix must be row-stochastic. "
            f"Maximum row-sum deviation from 1: {max_deviation:.3e}."
        )

    pi_vec = np.asarray(pi, dtype=float).reshape(-1)

    if pi_vec.shape[0] != n_states:
        raise ValueError(
            "Stationary distribution size must match the transition matrix. "
            f"Expected {n_states}, got {pi_vec.shape[0]}."
        )

    if not np.all(np.isfinite(pi_vec)):
        raise ValueError("Stationary distribution contains NaN or infinite values.")

    if np.any(pi_vec < -atol):
        min_value = float(np.min(pi_vec))
        raise ValueError(
            f"Stationary distribution contains negative probabilities. "
            f"Minimum value: {min_value:.3e}."
        )

    pi_sum = float(pi_vec.sum())

    if not np.isclose(pi_sum, 1.0, atol=atol, rtol=rtol):
        raise ValueError(
            "Stationary distribution must sum to 1. " f"Observed sum: {pi_sum:.12g}."
        )

    stationary_residual = pi_vec @ T_arr - pi_vec

    if not np.allclose(
        pi_vec @ T_arr,
        pi_vec,
        atol=stationary_atol,
        rtol=rtol,
    ):
        max_residual = float(np.max(np.abs(stationary_residual)))
        raise ValueError(
            "Provided pi is not stationary for the transition matrix. "
            f"Maximum absolute residual in pi @ T - pi: {max_residual:.3e}."
        )

    try:
        from deeptime.markov import pcca
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "PCCA+ membership computation requires the 'deeptime' package."
        ) from exc

    model = pcca(
        T_arr,
        n_metastable,
        stationary_distribution=pi_vec,
    )

    memberships = np.asarray(model.memberships, dtype=float)

    expected_shape = (n_states, n_metastable)

    if memberships.shape != expected_shape:
        raise ValueError(
            "PCCA+ returned memberships with unexpected shape. "
            f"Expected {expected_shape}, got {memberships.shape}."
        )

    if not np.all(np.isfinite(memberships)):
        raise ValueError("PCCA+ returned NaN or infinite membership values.")

    if np.any(memberships < -atol):
        min_value = float(np.min(memberships))
        raise ValueError(
            "PCCA+ returned negative membership probabilities. "
            f"Minimum value: {min_value:.3e}."
        )

    membership_row_sums = memberships.sum(axis=1)

    if not np.allclose(membership_row_sums, 1.0, atol=atol, rtol=rtol):
        max_deviation = float(np.max(np.abs(membership_row_sums - 1.0)))
        raise ValueError(
            "PCCA+ memberships must sum to 1 for each microstate. "
            f"Maximum row-sum deviation from 1: {max_deviation:.3e}."
        )

    # Canonicalize macrostate order by stationary macrostate population.
    # This makes label 0 the most populated macrostate, label 1 the second most
    # populated, etc.
    macro_weights = pi_vec @ memberships

    # Deterministic tie-breaking:
    # primary key: decreasing macrostate weight
    # secondary key: original macrostate index
    order = np.lexsort((np.arange(n_metastable), -macro_weights))

    memberships = memberships[:, order]

    macrostate_labels = np.argmax(memberships, axis=1).astype(int)

    return memberships, macrostate_labels


def find_conformations(
    msm_data: Dict[str, Any],
    trajectories: Optional[Any] = None,
    source_states: Optional[np.ndarray] = None,
    sink_states: Optional[np.ndarray] = None,
    auto_detect: bool = True,
    auto_detect_method: str = "auto",
    find_transition_states: bool = True,
    find_metastable_states: bool = True,
    compute_kis: bool = True,
    uncertainty_analysis: bool = False,
    n_bootstrap: int = 100,
    representative_selection: str = "closest_to_centroid",
    output_dir: Optional[str] = None,
    save_structures: bool = False,
    tse_tolerance: float = 0.05,
    n_metastable: Optional[int] = None,
    temperature: float = 300.0,
    k_slow: Union[int, str] = "auto",
    n_paths: int = 5,
    lag: int = 1,
    random_seed: int = 42,
    topology_path: Optional[str] = None,
    trajectory_locator: Optional[TrajectoryFrameLocator] = None,
) -> ConformationSet:
    """Find protein conformations using Transition Path Theory.

    This is the main high-level API for comprehensive conformational analysis.

    Args:
        msm_data: Dictionary containing:
            - 'T': Transition matrix (required)
            - 'pi': Stationary distribution (required)
            - 'dtrajs': Discrete trajectories (required for uncertainty)
            - 'features': Feature matrix (required for representatives)
            - 'fes': Free energy surface (optional, for auto-detection)
            - 'its': Implied timescales (optional, for auto-detection)
        trajectories: MDTraj trajectory or list of trajectories (for structure extraction)
        source_states: Source state indices (if not auto-detected)
        sink_states: Sink state indices (if not auto-detected)
        auto_detect: Auto-detect source/sink states
        auto_detect_method: Detection method ('auto', 'fes', 'timescale', 'population')
        find_transition_states: Include reactive states outside the source/sink sets
        find_metastable_states: Include source and sink states
        compute_kis: Compute Kinetic Importance Score
        uncertainty_analysis: Perform bootstrap uncertainty quantification
        n_bootstrap: Number of bootstrap samples
        representative_selection: Method for picking representatives ('closest_to_centroid',
            'centroid', 'true_medoid', 'diverse')
        output_dir: Directory for saving structures
        save_structures: Save representative structures as PDB files
        tse_tolerance: Committor tolerance from 0.5 used to classify transition state ensemble members
        n_metastable: Number of macrostates for PCCA+ (default: 2)
        temperature: Simulation temperature in Kelvin (default: 300.0)
        k_slow: Number of slow processes for KIS; 'auto' for automatic detection
        n_paths: Number of TPT pathways to extract (default: 5)
        lag: Lag time in frames for bootstrap MSM estimation (default: 1)
        random_seed: Random seed for bootstrap sampling (default: 42)
        topology_path: Path to topology file for structure extraction
        trajectory_locator: Custom locator for mapping frames to trajectory files

    Returns:
        ConformationSet with all identified conformations

    Example:
        >>> from pmarlo.api import find_conformations_from_msm
        >>> msm_data = {'T': T, 'pi': pi, 'dtrajs': dtrajs, 'features': features}
        >>> results = find_conformations_from_msm(
        ...     msm_data, trajectories=traj, compute_kis=True
        ... )
        >>> ts_conformations = results.get_transition_states()
        >>> print(f"Found {len(ts_conformations)} transition states")
    """
    logger.info("Starting conformations finder with TPT analysis")

    if "T" not in msm_data or "pi" not in msm_data:
        raise ValueError(
            "msm_data must contain 'T' (transition matrix) and 'pi' (stationary distribution)"
        )

    if not (0.0 <= tse_tolerance <= 0.5):
        raise ValueError("tse_tolerance must be between 0 and 0.5 inclusive")

    T = np.asarray(msm_data["T"])
    pi = np.asarray(msm_data["pi"])
    dtrajs = msm_data.get("dtrajs")
    features = msm_data.get("features")
    fes = msm_data.get("fes")
    its = msm_data.get("its")

    n_states = T.shape[0]
    n_metastable_resolved = _resolve_n_metastable(n_metastable, n_states)

    conformations: List[Conformation] = []
    tpt_result: Optional[TPTResult] = None
    kis_result: Optional[KISResult] = None
    uncertainty_results: List[UncertaintyResult] = []
    macrostate_labels: Optional[np.ndarray] = None
    macrostate_memberships: Optional[np.ndarray] = None

    needs_macrostates = find_metastable_states or find_transition_states
    if needs_macrostates:
        try:
            macrostate_memberships, macrostate_labels = _compute_macrostate_memberships(
                T, pi, n_metastable_resolved
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to compute PCCA+ macrostates required for conformations analysis"
            ) from exc

    # Step 1: Detect or validate source/sink states
    if auto_detect:
        logger.info("Auto-detecting source and sink states")
        detector = StateDetector()
        source_states, sink_states = detector.auto_detect(
            T=T,
            pi=pi,
            fes=fes,
            its=its,
            n_states=n_metastable_resolved,
            method=auto_detect_method,
        )
        logger.info(
            f"Detected {len(source_states)} source states and {len(sink_states)} sink states"
        )
    elif source_states is None or sink_states is None:
        raise ValueError(
            "source_states and sink_states must be provided when auto_detect=False"
        )
    else:
        source_states = np.asarray(source_states)
        sink_states = np.asarray(sink_states)
        logger.info("Using provided source and sink states")

    # Step 2: Run TPT analysis
    logger.info("Running Transition Path Theory analysis")
    tpt = TPTAnalysis(T, pi)
    tpt_result = tpt.analyze(source_states, sink_states, n_paths=n_paths)

    # Step 3: Compute KIS if requested
    if compute_kis:
        logger.info("Computing Kinetic Importance Scores")
        kis_calc = KineticImportanceScore(T, pi)
        kis_result = kis_calc.compute(k_slow=k_slow, its=its)

        if uncertainty_analysis and dtrajs is not None:
            logger.info("Computing KIS stability")
            stability, boot_std = kis_calc.bootstrap_stability(
                dtrajs,
                n_boot=n_bootstrap // 2,
                top_n=10,
                random_seed=random_seed,
                lag=lag,
            )
            kis_result = KISResult(
                kis_scores=kis_result.kis_scores,
                k_slow=kis_result.k_slow,
                eigenvectors=kis_result.eigenvectors,
                eigenvalues=kis_result.eigenvalues,
                ranked_states=kis_result.ranked_states,
                stability_metric=stability,
                bootstrap_std=boot_std,
            )

    # Step 4: Identify metastable and transition states from TPT results
    flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

    if find_metastable_states:
        logger.info("Classifying metastable states from source and sink sets")
        metastable_conformations = _find_metastable_states(
            tpt_result,
            pi,
            temperature,
            kis_result,
            flux_by_state,
            macrostate_labels,
            macrostate_memberships,
        )
        conformations.extend(metastable_conformations)

    if find_transition_states:
        logger.info(
            "Classifying reactive (transition) states " f"(tolerance={tse_tolerance})"
        )
        transition_conformations = _find_transition_states(
            tpt_result,
            pi,
            temperature,
            kis_result,
            flux_by_state,
            macrostate_labels,
            tse_tolerance=tse_tolerance,
        )
        conformations.extend(transition_conformations)

    # Step 5: Select representative structures
    if features is not None and dtrajs is not None and len(conformations) > 0:
        logger.info(
            f"Selecting representatives using {representative_selection} method"
        )
        picker = RepresentativePicker()
        state_ids = list(set(c.state_id for c in conformations))
        representatives = picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=state_ids,
            n_reps=1,
            method=representative_selection,
        )
        _update_with_representatives(conformations, representatives)

    # Step 6: Extract and save structures
    if save_structures and output_dir is not None:
        logger.info("Extracting representative structures")
        _extract_structures(
            conformations,
            trajectories,
            output_dir,
            topology_path=topology_path,
            trajectory_locator=trajectory_locator,
        )

    # Step 7: Uncertainty quantification
    if uncertainty_analysis and dtrajs is not None:
        logger.info("Performing uncertainty quantification")
        quantifier = UncertaintyQuantifier(random_seed=random_seed)
        tpt_uncertainties = quantifier.bootstrap_tpt(
            dtrajs,
            source_states,
            sink_states,
            n_boot=n_bootstrap,
            lag=lag,
        )
        uncertainty_results.extend(tpt_uncertainties.values())
        fe_uncertainty = quantifier.bootstrap_free_energies(
            dtrajs, T_K=temperature, n_boot=n_bootstrap
        )
        uncertainty_results.append(fe_uncertainty)

    metastable_count = sum(
        1 for c in conformations if c.conformation_type == "metastable"
    )
    transition_count = sum(
        1 for c in conformations if c.conformation_type == "transition"
    )
    tse_count = sum(1 for c in conformations if c.conformation_type == "tse")

    result = ConformationSet(
        conformations=conformations,
        tpt_result=tpt_result,
        kis_result=kis_result,
        uncertainty_results=uncertainty_results,
        macrostate_labels=macrostate_labels,
        metadata={
            "n_conformations": len(conformations),
            "auto_detected": auto_detect,
            "temperature_K": temperature,
            "uncertainty_analysis": uncertainty_analysis,
            "n_metastable_states": n_metastable_resolved,
            "n_transition_state_ensemble": tse_count,
        },
    )

    logger.info(
        f"Conformations finder complete: found {len(conformations)} conformations "
        f"({metastable_count} metastable, "
        f"{transition_count} transition, "
        f"{tse_count} transition state ensemble)"
    )

    return result


def _find_transition_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: Optional[np.ndarray] = None,
    macrostate_labels: Optional[np.ndarray] = None,
    tse_tolerance: float = 0.05,
) -> List[Conformation]:
    """Identify all reactive (non-source/sink) states."""

    kT = kT_kJ_per_mol(temperature_K)
    source_states = set(int(s) for s in np.asarray(tpt_result.source_states))
    sink_states = set(int(s) for s in np.asarray(tpt_result.sink_states))

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

    conformations = []

    for state_id in range(len(pi)):
        if state_id in source_states or state_id in sink_states:
            continue

        population = float(pi[state_id])
        committor = float(tpt_result.forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id])

        macrostate_id = None
        if macrostate_labels is not None and state_id < macrostate_labels.shape[0]:
            macrostate_id = int(macrostate_labels[state_id])

        is_tse = abs(committor - 0.5) <= tse_tolerance
        conformations.append(
            Conformation(
                conformation_type="tse" if is_tse else "transition",
                state_id=int(state_id),
                population=population,
                free_energy=-kT * np.log(population) if population > 0 else np.inf,
                committor=committor,
                kis_score=kis_score,
                flux=flux,
                macrostate_id=macrostate_id,
            )
        )

    return conformations


def _find_metastable_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: Optional[np.ndarray] = None,
    macrostate_labels: Optional[np.ndarray] = None,
    macrostate_memberships: Optional[np.ndarray] = None,
) -> List[Conformation]:
    """Identify metastable states as source and sink sets from TPT results."""
    kT = kT_kJ_per_mol(temperature_K)

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

    source_states = [int(s) for s in np.asarray(tpt_result.source_states)]
    sink_states = [int(s) for s in np.asarray(tpt_result.sink_states)]
    metastable_states = sorted(set(source_states + sink_states))

    conformations: List[Conformation] = []

    for state_id in metastable_states:
        population = float(pi[state_id])
        committor = float(tpt_result.forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id])

        macrostate_id = None
        if macrostate_labels is not None and state_id < macrostate_labels.shape[0]:
            macrostate_id = int(macrostate_labels[state_id])

        conf_metadata: Dict[str, Any] = {
            "role": "source" if state_id in source_states else "sink"
        }
        if (
            macrostate_memberships is not None
            and state_id < macrostate_memberships.shape[0]
        ):
            conf_metadata["macrostate_members"] = macrostate_memberships[state_id]

        conformations.append(
            Conformation(
                conformation_type="metastable",
                state_id=int(state_id),
                population=population,
                free_energy=-kT * np.log(population) if population > 0 else np.inf,
                committor=committor,
                kis_score=kis_score,
                flux=flux,
                macrostate_id=macrostate_id,
                metadata=conf_metadata,
            )
        )

    return conformations


def _calculate_state_flux(flux_matrix: np.ndarray) -> np.ndarray:
    """Compute the total reactive flux through each state."""
    return 0.5 * (np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0))


def _update_with_representatives(
    conformations: List[Conformation],
    representatives: List[Tuple[int, int, int, int]],
) -> None:
    """Update conformations in-place with representative frame information."""
    rep_dict = {
        state_id: (frame_idx, traj_idx, local_idx)
        for state_id, frame_idx, traj_idx, local_idx in representatives
    }

    for conf in conformations:
        if conf.state_id in rep_dict:
            frame_idx, traj_idx, local_idx = rep_dict[conf.state_id]
            conf.frame_index = frame_idx
            conf.trajectory_index = traj_idx
            conf.local_frame_index = local_idx


def _extract_structures(
    conformations: List[Conformation],
    trajectories: Any,
    output_dir: str,
    *,
    topology_path: Optional[str] = None,
    trajectory_locator: Optional[TrajectoryFrameLocator] = None,
) -> None:
    """Extract and save representative structures."""
    picker = RepresentativePicker()

    for conf_type in ["metastable", "transition", "tse"]:
        type_conformations = [
            c for c in conformations if c.conformation_type == conf_type
        ]

        if not type_conformations:
            continue

        representatives: List[Tuple[int, int, int, int]] = []
        for c in type_conformations:
            if c.frame_index is None:
                continue
            if c.trajectory_index is None or c.local_frame_index is None:
                raise ValueError(
                    "Representative selection missing trajectory or local frame index "
                    f"for state {c.state_id}"
                )
            representatives.append(
                (c.state_id, c.frame_index, c.trajectory_index, c.local_frame_index)
            )

        if not representatives:
            continue

        type_dir = str(Path(output_dir) / conf_type)
        saved_paths = picker.extract_structures(
            representatives,
            trajectories,
            type_dir,
            prefix=conf_type,
            topology_path=topology_path,
            trajectory_locator=trajectory_locator,
        )

        for i, conf in enumerate(type_conformations):
            if i < len(saved_paths):
                conf.structure_path = saved_paths[i]
