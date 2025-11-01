"""High-level API for finding conformations using TPT analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import constants

from .kinetic_importance import KineticImportanceScore
from .representative_picker import RepresentativePicker, build_frame_index_lookup
from .results import Conformation, ConformationSet, KISResult, TPTResult, UncertaintyResult
from .state_detection import StateDetector
from .tpt_analysis import TPTAnalysis
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger("pmarlo.conformations")


def find_conformations(
    msm_data: Dict[str, Any],
    trajectories: Optional[Any] = None,
    source_states: Optional[np.ndarray] = None,
    sink_states: Optional[np.ndarray] = None,
    auto_detect: bool = True,
    auto_detect_method: str = "auto",
    find_transition_states: bool = True,
    find_metastable_states: bool = True,
    find_pathway_intermediates: bool = True,
    compute_kis: bool = True,
    uncertainty_analysis: bool = False,
    n_bootstrap: int = 100,
    representative_selection: str = "medoid",
    output_dir: Optional[str] = None,
    save_structures: bool = False,
    **kwargs: Any,
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
        find_pathway_intermediates: Alias for including reactive states (for backward compatibility)
        compute_kis: Compute Kinetic Importance Score
        uncertainty_analysis: Perform bootstrap uncertainty quantification
        n_bootstrap: Number of bootstrap samples
        representative_selection: Method for picking representatives ('medoid', 'centroid', 'diverse')
        output_dir: Directory for saving structures
        save_structures: Save representative structures as PDB files
        **kwargs: Additional options

    Returns:
        ConformationSet with all identified conformations

    Example:
        >>> msm_data = {'T': T, 'pi': pi, 'dtrajs': dtrajs, 'features': features}
        >>> results = find_conformations(msm_data, trajectories=traj, compute_kis=True)
        >>> ts_conformations = results.get_transition_states()
        >>> print(f"Found {len(ts_conformations)} transition states")
    """
    logger.info("Starting conformations finder with TPT analysis")

    # Validate required inputs
    if "T" not in msm_data or "pi" not in msm_data:
        raise ValueError("msm_data must contain 'T' (transition matrix) and 'pi' (stationary distribution)")

    T = np.asarray(msm_data["T"])
    pi = np.asarray(msm_data["pi"])
    dtrajs = msm_data.get("dtrajs")
    features = msm_data.get("features")
    fes = msm_data.get("fes")
    its = msm_data.get("its")

    n_states = T.shape[0]
    temperature_K = kwargs.get("temperature", 300.0)

    # Initialize result containers
    conformations: List[Conformation] = []
    tpt_result: Optional[TPTResult] = None
    kis_result: Optional[KISResult] = None
    uncertainty_results: List[UncertaintyResult] = []
    macrostate_labels: Optional[np.ndarray] = None

    # Step 1: Detect or validate source/sink states
    if auto_detect or source_states is None or sink_states is None:
        logger.info("Auto-detecting source and sink states")
        detector = StateDetector()
        source_states, sink_states = detector.auto_detect(
            T=T,
            pi=pi,
            fes=fes,
            its=its,
            n_states=kwargs.get("n_metastable", 2),
            method=auto_detect_method,
        )
        logger.info(
            f"Detected {len(source_states)} source states and {len(sink_states)} sink states"
        )
    else:
        source_states = np.asarray(source_states)
        sink_states = np.asarray(sink_states)
        logger.info("Using provided source and sink states")

    # Step 2: Run TPT analysis
    logger.info("Running Transition Path Theory analysis")
    tpt = TPTAnalysis(T, pi)
    tpt_result = tpt.analyze(source_states, sink_states, n_paths=kwargs.get("n_paths", 5))

    # Step 3: Compute KIS if requested
    if compute_kis:
        logger.info("Computing Kinetic Importance Scores")
        kis_calc = KineticImportanceScore(T, pi)
        k_slow = kwargs.get("k_slow", "auto")
        kis_result = kis_calc.compute(k_slow=k_slow, its=its)

        # Optional: KIS stability analysis
        if uncertainty_analysis and dtrajs is not None:
            logger.info("Computing KIS stability")
            stability, boot_std = kis_calc.bootstrap_stability(
                dtrajs, n_boot=n_bootstrap // 2, top_n=10
            )
            # Update kis_result with stability info
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
    if tpt_result is not None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

        if find_metastable_states:
            logger.info("Classifying metastable states from source and sink sets")
            macrostate_labels, metastable_conformations = _find_metastable_states(
                tpt_result,
                pi,
                temperature_K,
                kis_result,
                flux_by_state,
            )
            conformations.extend(metastable_conformations)

        if find_transition_states or find_pathway_intermediates:
            logger.info("Classifying reactive (transition) states")
            transition_conformations = _find_transition_states(
                tpt_result,
                pi,
                temperature_K,
                kis_result,
                flux_by_state,
            )
            conformations.extend(transition_conformations)

    # Step 5: Select representative structures
    if features is not None and dtrajs is not None and len(conformations) > 0:
        logger.info(f"Selecting representatives using {representative_selection} method")
        picker = RepresentativePicker()

        # Get unique states from conformations
        state_ids = list(set(c.state_id for c in conformations))
        weights = msm_data.get("weights")  # TRAM/MBAR weights if available

        representatives = picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=state_ids,
            weights=weights,
            n_reps=1,
            method=representative_selection,
        )

        # Update conformations with representative info
        _update_with_representatives(conformations, representatives)

    # Step 6: Extract and save structures
    if save_structures and trajectories is not None and output_dir is not None:
        logger.info("Extracting representative structures")
        _extract_structures(conformations, trajectories, output_dir)

    # Step 7: Uncertainty quantification
    if uncertainty_analysis and dtrajs is not None:
        logger.info("Performing uncertainty quantification")
        quantifier = UncertaintyQuantifier(random_seed=kwargs.get("random_seed", 42))

        # Bootstrap TPT observables
        tpt_uncertainties = quantifier.bootstrap_tpt(
            dtrajs, source_states, sink_states, n_boot=n_bootstrap, lag=kwargs.get("lag", 1)
        )
        uncertainty_results.extend(tpt_uncertainties.values())

        # Bootstrap free energies
        fe_uncertainty = quantifier.bootstrap_free_energies(
            dtrajs, T_K=temperature_K, n_boot=n_bootstrap
        )
        uncertainty_results.append(fe_uncertainty)

    # Assemble final results
    metastable_count = sum(1 for c in conformations if c.conformation_type == "metastable")
    transition_count = sum(1 for c in conformations if c.conformation_type == "transition")
    pathway_count = sum(1 for c in conformations if c.conformation_type == "pathway")

    metadata = {
        "n_conformations": len(conformations),
        "n_pathway_intermediates": pathway_count,
        "auto_detected": auto_detect,
        "temperature_K": temperature_K,
        "uncertainty_analysis": uncertainty_analysis,
    }

    result = ConformationSet(
        conformations=conformations,
        tpt_result=tpt_result,
        kis_result=kis_result,
        uncertainty_results=uncertainty_results,
        macrostate_labels=macrostate_labels,
        metadata=metadata,
    )

    logger.info(
        f"Conformations finder complete: found {len(conformations)} conformations "
        f"({metastable_count} metastable, "
        f"{transition_count} transition, "
        f"{pathway_count} pathway)"
    )

    return result


def _find_transition_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: Optional[np.ndarray] = None,
) -> List[Conformation]:
    """Identify all reactive (non-source/sink) states."""
    kT = constants.k * temperature_K * constants.Avogadro / 1000.0
    source_states = set(int(s) for s in np.asarray(tpt_result.source_states))
    sink_states = set(int(s) for s in np.asarray(tpt_result.sink_states))

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

    conformations = []

    for state_id in range(len(pi)):
        if state_id in source_states or state_id in sink_states:
            continue

        population = float(pi[state_id])
        free_energy = -kT * np.log(max(population, 1e-10))
        committor = float(tpt_result.forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id])

        conf = Conformation(
            conformation_type="transition",
            state_id=int(state_id),
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            committor=committor,
            kis_score=kis_score,
            flux=flux,
        )

        conformations.append(conf)

    return conformations


def _find_metastable_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[Conformation]]:
    """Identify metastable states as source and sink sets from TPT results."""
    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

    source_states = [int(s) for s in np.asarray(tpt_result.source_states)]
    sink_states = [int(s) for s in np.asarray(tpt_result.sink_states)]
    metastable_states = sorted(set(source_states + sink_states))

    conformations: List[Conformation] = []

    for state_id in metastable_states:
        population = float(pi[state_id])
        free_energy = -kT * np.log(max(population, 1e-10))
        committor = float(tpt_result.forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id])

        if state_id in source_states:
            role = "source"
        else:
            role = "sink"

        conf = Conformation(
            conformation_type="metastable",
            state_id=int(state_id),
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            committor=committor,
            kis_score=kis_score,
            flux=flux,
            metadata={"role": role},
        )

        conformations.append(conf)

    return None, conformations


def _calculate_state_flux(flux_matrix: np.ndarray) -> np.ndarray:
    """Compute the total reactive flux through each state."""
    return np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)



def _assign_frame_indices(
    conformations: List[Conformation],
    dtrajs: List[np.ndarray],
    features: np.ndarray,
) -> None:
    """Assign frame indices to conformations."""

    lookup = build_frame_index_lookup(dtrajs)

    if features.shape[0] != lookup.n_frames:
        raise ValueError(
            "Feature matrix row count does not match total number of frames "
            f"({features.shape[0]} != {lookup.n_frames})."
        )

    for conf in conformations:
        frames_in_state = lookup.frames_for_state(conf.state_id)

        if frames_in_state.size == 0:
            raise ValueError(
                f"No frames available for conformation state {conf.state_id}"
            )

        state_features = features[frames_in_state]
        centroid = np.mean(state_features, axis=0)
        distances = np.linalg.norm(state_features - centroid, axis=1)
        best_local_idx = int(np.argmin(distances))
        global_frame = int(frames_in_state[best_local_idx])

        traj_idx, local_idx = lookup.to_local_indices(global_frame)

        conf.frame_index = global_frame
        conf.trajectory_index = int(traj_idx)
        conf.local_frame_index = int(local_idx)


def _update_with_representatives(
    conformations: List[Conformation],
    representatives: List[Tuple[int, int, int, int]],
) -> None:
    """Update conformations with representative frame information."""

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
) -> None:
    """Extract and save representative structures."""
    picker = RepresentativePicker()

    # Group by conformation type
    for conf_type in ["metastable", "transition", "pathway"]:
        type_conformations = [c for c in conformations if c.conformation_type == conf_type]

        if not type_conformations:
            continue

        # Build representatives list
        representatives: List[Tuple[int, int, int, int]] = []
        for c in type_conformations:
            if c.frame_index < 0:
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

        # Save structures
        type_dir = str(Path(output_dir) / conf_type)
        saved_paths = picker.extract_structures(
            representatives, trajectories, type_dir, prefix=conf_type
        )

        # Update conformations with paths
        for i, conf in enumerate(type_conformations):
            if i < len(saved_paths):
                conf.structure_path = saved_paths[i]

