"""High-level API for finding conformations using TPT analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import constants

from .kinetic_importance import KineticImportanceScore
from .representative_picker import RepresentativePicker
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
        find_transition_states: Identify transition state ensemble
        find_metastable_states: Identify metastable states
        find_pathway_intermediates: Identify pathway intermediates
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

    # Step 4: Identify metastable states
    if find_metastable_states:
        logger.info("Identifying metastable states")
        n_metastable = kwargs.get("n_metastable", 4)
        macrostate_labels, macro_conformations = _find_metastable_states(
            T, pi, n_metastable, temperature_K, kis_result
        )

        conformations.extend(macro_conformations)

    # Step 5: Identify transition states
    if find_transition_states and tpt_result is not None:
        logger.info("Identifying transition state ensemble")
        ts_conformations = _find_transition_states(
            tpt_result, pi, temperature_K, kis_result
        )
        conformations.extend(ts_conformations)

    # Step 6: Identify pathway intermediates
    if find_pathway_intermediates and tpt_result is not None:
        logger.info("Identifying pathway intermediates")
        pathway_conformations = _find_pathway_intermediates(
            tpt_result, pi, temperature_K, kis_result
        )
        conformations.extend(pathway_conformations)

    # Step 7: Map conformations to frames (if features available)
    if features is not None and dtrajs is not None:
        logger.info("Mapping conformations to frames")
        _assign_frame_indices(conformations, dtrajs, features)

    # Step 8: Select representative structures
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

    # Step 9: Extract and save structures
    if save_structures and trajectories is not None and output_dir is not None:
        logger.info("Extracting representative structures")
        _extract_structures(conformations, trajectories, output_dir)

    # Step 10: Uncertainty quantification
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
    metadata = {
        "n_conformations": len(conformations),
        "n_transition_states": len([c for c in conformations if c.conformation_type == "transition"]),
        "n_metastable_states": len([c for c in conformations if c.conformation_type == "metastable"]),
        "n_pathway_intermediates": len([c for c in conformations if c.conformation_type == "pathway"]),
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
        f"({metadata['n_metastable_states']} metastable, "
        f"{metadata['n_transition_states']} transition, "
        f"{metadata['n_pathway_intermediates']} pathway)"
    )

    return result


def _find_metastable_states(
    T: np.ndarray,
    pi: np.ndarray,
    n_metastable: int,
    temperature_K: float,
    kis_result: Optional[KISResult],
) -> Tuple[np.ndarray, List[Conformation]]:
    """Identify metastable states using PCCA+."""
    try:
        from deeptime.markov import pcca
    except ImportError:
        logger.warning("PCCA+ requires deeptime, skipping metastable states")
        return np.array([]), []

    try:
        model = pcca(T, n_metastable)
        memberships = np.asarray(model.memberships)
        labels = np.argmax(memberships, axis=1)
    except Exception as e:
        logger.warning(f"PCCA+ failed: {e}")
        return np.array([]), []

    # Compute macrostate populations and free energies
    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    conformations = []

    for macro_id in range(n_metastable):
        states_in_macro = np.where(labels == macro_id)[0]

        if len(states_in_macro) == 0:
            continue

        # Macrostate population
        macro_pop = float(np.sum(pi[states_in_macro]))

        # Free energy
        free_energy = -kT * np.log(max(macro_pop, 1e-10))

        # Select representative microstate (highest population)
        micro_pops = pi[states_in_macro]
        rep_local_idx = int(np.argmax(micro_pops))
        rep_state_id = int(states_in_macro[rep_local_idx])

        # KIS score if available
        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[rep_state_id])

        conf = Conformation(
            conformation_type="metastable",
            state_id=rep_state_id,
            macrostate_id=macro_id,
            frame_index=-1,  # Will be assigned later
            population=macro_pop,
            free_energy=free_energy,
            kis_score=kis_score,
            metadata={
                "n_microstates": len(states_in_macro),
                "microstate_ids": states_in_macro.tolist(),
            },
        )

        conformations.append(conf)

    return labels, conformations


def _find_transition_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
) -> List[Conformation]:
    """Identify transition state ensemble from committors."""
    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    # States with committor ~ 0.5
    ts_states = np.where(
        (tpt_result.forward_committor >= 0.4) & (tpt_result.forward_committor <= 0.6)
    )[0]

    conformations = []

    for state_id in ts_states:
        state_id_int = int(state_id)
        population = float(pi[state_id_int])
        free_energy = -kT * np.log(max(population, 1e-10))
        committor = float(tpt_result.forward_committor[state_id_int])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id_int])

        conf = Conformation(
            conformation_type="transition",
            state_id=state_id_int,
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            committor=committor,
            kis_score=kis_score,
        )

        conformations.append(conf)

    return conformations


def _find_pathway_intermediates(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
) -> List[Conformation]:
    """Identify pathway intermediates from reactive flux."""
    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    # Find states with high reactive flux (bottlenecks)
    flux_through_state = np.sum(tpt_result.flux_matrix, axis=1) + np.sum(
        tpt_result.flux_matrix, axis=0
    )

    # Take top states
    top_n = min(10, len(flux_through_state))
    bottleneck_states = np.argsort(flux_through_state)[::-1][:top_n]

    conformations = []

    for state_id in bottleneck_states:
        state_id_int = int(state_id)

        # Skip source/sink states
        if (
            state_id_int in tpt_result.source_states
            or state_id_int in tpt_result.sink_states
        ):
            continue

        population = float(pi[state_id_int])
        free_energy = -kT * np.log(max(population, 1e-10))
        flux = float(flux_through_state[state_id_int])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id_int])

        conf = Conformation(
            conformation_type="pathway",
            state_id=state_id_int,
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            flux=flux,
            kis_score=kis_score,
        )

        conformations.append(conf)

    return conformations


def _assign_frame_indices(
    conformations: List[Conformation],
    dtrajs: List[np.ndarray],
    features: np.ndarray,
) -> None:
    """Assign frame indices to conformations."""
    concatenated_dtrajs = np.concatenate(dtrajs)

    for conf in conformations:
        # Find frames in this state
        frames_in_state = np.where(concatenated_dtrajs == conf.state_id)[0]

        if len(frames_in_state) > 0:
            # Pick frame closest to centroid
            state_features = features[frames_in_state]
            centroid = np.mean(state_features, axis=0)
            distances = np.linalg.norm(state_features - centroid, axis=1)
            best_local_idx = int(np.argmin(distances))
            conf.frame_index = int(frames_in_state[best_local_idx])


def _update_with_representatives(
    conformations: List[Conformation],
    representatives: List[Tuple[int, int, Optional[int]]],
) -> None:
    """Update conformations with representative frame information."""
    rep_dict = {state_id: (frame_idx, traj_idx) for state_id, frame_idx, traj_idx in representatives}

    for conf in conformations:
        if conf.state_id in rep_dict:
            frame_idx, traj_idx = rep_dict[conf.state_id]
            conf.frame_index = frame_idx
            conf.trajectory_index = traj_idx


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
        representatives = [
            (c.state_id, c.frame_index, c.trajectory_index)
            for c in type_conformations
            if c.frame_index >= 0
        ]

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

