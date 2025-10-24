"""High-level API for finding conformations using TPT analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import constants

from .kinetic_importance import KineticImportanceScore
from .representative_picker import (
    RepresentativePicker,
    TrajectoryFrameLocator,
    build_frame_index_lookup,
)
from .results import (
    Conformation,
    ConformationSet,
    KISResult,
    TPTResult,
    UncertaintyResult,
)
from .tpt_analysis import TPTAnalysis
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger("pmarlo.conformations")


def find_conformations(
    msm_data: Dict[str, Any],
    trajectories: Optional[Any] = None,
    trajectory_locator: TrajectoryFrameLocator | None = None,
    topology_path: str | Path | None = None,
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
    tse_tolerance: float = 0.1,
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
        trajectory_locator: Resolver mapping global frame indices to trajectory files
        topology_path: Topology PDB used with ``trajectory_locator`` when loading frames
        source_states: Source state indices (if not auto-detected)
        sink_states: Sink state indices (if not auto-detected)
        auto_detect: Auto-detect source/sink states
        auto_detect_method: Detection method ('auto', 'fes', 'timescale', 'population')
        find_transition_states: Include reactive states outside the source/sink macrostates
        find_metastable_states: Include PCCA+ macrostates as conformations
        find_pathway_intermediates: Alias for including reactive states (for backward compatibility)
        compute_kis: Compute Kinetic Importance Score
        uncertainty_analysis: Perform bootstrap uncertainty quantification
        n_bootstrap: Number of bootstrap samples
        representative_selection: Method for picking representatives ('medoid', 'centroid', 'diverse')
        tse_tolerance: Allowed deviation from 0.5 when identifying the transition state ensemble
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
        raise ValueError(
            "msm_data must contain 'T' (transition matrix) and 'pi' (stationary distribution)"
        )

    T = np.asarray(msm_data["T"])
    pi = np.asarray(msm_data["pi"])
    dtrajs = msm_data.get("dtrajs")
    features = msm_data.get("features")
    its = msm_data.get("its")
    temperature_K = kwargs.get("temperature", 300.0)

    # Initialize result containers
    conformations: List[Conformation] = []
    tpt_result: Optional[TPTResult] = None
    kis_result: Optional[KISResult] = None
    uncertainty_results: List[UncertaintyResult] = []

    # Step 1: Lump microstates into metastable macrostates with PCCA+
    n_macrostates = int(kwargs.get("n_metastable", 2))
    (
        macrostate_labels,
        macrostate_memberships,
        macrostate_sets,
        macrostate_populations,
    ) = _compute_pcca_macrostates(T, pi, n_macrostates)
    macrostate_roles: Dict[str, Set[int]] = {"source": set(), "sink": set()}

    # Step 2: Detect or validate source/sink states
    if source_states is None or sink_states is None:
        if not auto_detect:
            raise ValueError(
                "source_states and sink_states must be provided when auto_detect is False"
            )

        logger.info("Auto-detecting source and sink macrostates from PCCA+ populations")
        source_macro_id, sink_macro_id = _select_source_sink_macrostates(
            macrostate_populations
        )
        source_states = macrostate_sets[source_macro_id]
        sink_states = macrostate_sets[sink_macro_id]
        macrostate_roles["source"].add(int(source_macro_id))
        macrostate_roles["sink"].add(int(sink_macro_id))
        logger.info(
            "Selected macrostates %d (|A|=%d) and %d (|B|=%d) as source/sink",
            source_macro_id,
            len(source_states),
            sink_macro_id,
            len(sink_states),
        )
    else:
        source_states = np.asarray(source_states, dtype=int)
        sink_states = np.asarray(sink_states, dtype=int)
        if source_states.size == 0 or sink_states.size == 0:
            raise ValueError("source_states and sink_states must be non-empty")

        logger.info("Using provided source and sink state indices")
        macrostate_roles["source"].update(
            int(macro_id) for macro_id in np.unique(macrostate_labels[source_states])
        )
        macrostate_roles["sink"].update(
            int(macro_id) for macro_id in np.unique(macrostate_labels[sink_states])
        )

    # Step 3: Run TPT analysis
    logger.info("Running Transition Path Theory analysis")
    tpt = TPTAnalysis(T, pi)
    tpt_result = tpt.analyze(
        source_states, sink_states, n_paths=kwargs.get("n_paths", 5)
    )

    # Step 4: Compute KIS if requested
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

    # Step 5: Identify metastable macrostates and microstate classifications
    if tpt_result is not None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)

        if find_metastable_states:
            logger.info("Creating metastable conformations from PCCA+ macrostates")
            metastable_conformations = _create_metastable_conformations(
                macrostate_sets,
                macrostate_populations,
                macrostate_roles,
                pi,
                temperature_K,
                tpt_result,
                flux_by_state,
                kis_result,
            )
            conformations.extend(metastable_conformations)

        if find_transition_states or find_pathway_intermediates:
            logger.info(
                "Classifying reactive microstates and transition state ensemble"
            )
            tse_state_ids = set(
                tpt.identify_transition_state_ensemble(
                    tpt_result.forward_committor, tolerance=tse_tolerance
                )
            )
            transition_conformations = _create_transition_conformations(
                tpt_result,
                pi,
                temperature_K,
                kis_result,
                flux_by_state,
                macrostate_labels,
                macrostate_memberships,
                macrostate_roles,
                tse_state_ids,
            )
            conformations.extend(transition_conformations)

    # Step 5: Select representative structures
    if features is not None and dtrajs is not None and len(conformations) > 0:
        logger.info(
            f"Selecting representatives using {representative_selection} method"
        )
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
    if save_structures and output_dir is not None:
        if trajectory_locator is None and trajectories is None:
            raise ValueError(
                "Either trajectories or trajectory_locator must be provided when saving structures"
            )
        logger.info("Extracting representative structures")
        _extract_structures(
            conformations,
            trajectories,
            output_dir,
            trajectory_locator=trajectory_locator,
            topology_path=topology_path,
        )

    # Step 7: Uncertainty quantification
    if uncertainty_analysis and dtrajs is not None:
        logger.info("Performing uncertainty quantification")
        quantifier = UncertaintyQuantifier(random_seed=kwargs.get("random_seed", 42))

        # Bootstrap TPT observables
        tpt_uncertainties = quantifier.bootstrap_tpt(
            dtrajs,
            source_states,
            sink_states,
            n_boot=n_bootstrap,
            lag=kwargs.get("lag", 1),
        )
        uncertainty_results.extend(tpt_uncertainties.values())

        # Bootstrap free energies
        fe_uncertainty = quantifier.bootstrap_free_energies(
            dtrajs, T_K=temperature_K, n_boot=n_bootstrap
        )
        uncertainty_results.append(fe_uncertainty)

    # Assemble final results
    metastable_count = sum(
        1 for c in conformations if c.conformation_type == "metastable"
    )
    transition_count = sum(
        1 for c in conformations if c.conformation_type == "transition"
    )
    pathway_count = sum(1 for c in conformations if c.conformation_type == "pathway")
    tse_count = sum(1 for c in conformations if c.conformation_type == "tse")

    metadata = {
        "n_conformations": len(conformations),
        "n_pathway_intermediates": pathway_count,
        "n_transition_state_ensemble": tse_count,
        "auto_detected": auto_detect,
        "temperature_K": temperature_K,
        "uncertainty_analysis": uncertainty_analysis,
        "macrostate_populations": macrostate_populations.tolist(),
        "macrostate_roles": {
            "source": sorted(macrostate_roles["source"]),
            "sink": sorted(macrostate_roles["sink"]),
        },
        "tse_tolerance": tse_tolerance,
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
        f"{tse_count} tse, "
        f"{pathway_count} pathway)"
    )

    return result


def _calculate_state_flux(flux_matrix: np.ndarray) -> np.ndarray:
    """Compute the total reactive flux through each state."""
    return np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)


def _compute_pcca_macrostates(
    T: np.ndarray,
    pi: np.ndarray,
    n_macrostates: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Run PCCA+ to lump microstates into macrostates without fallbacks."""
    if n_macrostates < 2:
        raise ValueError("n_metastable (number of macrostates) must be at least 2")
    if n_macrostates > T.shape[0]:
        raise ValueError("n_metastable cannot exceed the number of microstates")

    try:
        from deeptime.markov import pcca as deeptime_pcca
    except ImportError as exc:  # pragma: no cover - tested indirectly
        raise ImportError("PCCA+ detection requires deeptime to be installed") from exc

    model = deeptime_pcca(T, n_macrostates)
    memberships = np.asarray(model.memberships, dtype=float)
    if memberships.ndim != 2 or memberships.shape[1] != n_macrostates:
        raise ValueError("PCCA+ returned memberships with unexpected shape")

    labels = np.argmax(memberships, axis=1)
    macrostate_sets: List[np.ndarray] = []
    for macro_id in range(n_macrostates):
        states = np.where(labels == macro_id)[0]
        if states.size == 0:
            raise ValueError(f"PCCA+ produced an empty macrostate (id={macro_id})")
        macrostate_sets.append(states.astype(int))

    macrostate_populations = np.array(
        [float(np.sum(pi[state_indices])) for state_indices in macrostate_sets],
        dtype=float,
    )

    return labels.astype(int), memberships, macrostate_sets, macrostate_populations


def _select_source_sink_macrostates(
    macrostate_populations: np.ndarray,
) -> Tuple[int, int]:
    """Select source and sink macrostates based on population ordering."""
    if macrostate_populations.size < 2:
        raise ValueError("At least two macrostates are required for TPT analysis")

    sorted_indices = np.argsort(macrostate_populations)[::-1]
    source_macro = int(sorted_indices[0])
    sink_macro: Optional[int] = None

    for idx in sorted_indices[1:]:
        if macrostate_populations[idx] > 0:
            sink_macro = int(idx)
            break

    if sink_macro is None:
        raise ValueError(
            "Unable to identify a sink macrostate with non-zero population"
        )

    if sink_macro == source_macro:
        raise ValueError("Source and sink macrostates must be distinct")

    return source_macro, sink_macro


def _create_metastable_conformations(
    macrostate_sets: Sequence[np.ndarray],
    macrostate_populations: np.ndarray,
    macrostate_roles: Dict[str, Set[int]],
    pi: np.ndarray,
    temperature_K: float,
    tpt_result: TPTResult,
    flux_by_state: np.ndarray,
    kis_result: Optional[KISResult],
) -> List[Conformation]:
    """Create Conformation objects for each PCCA+ macrostate."""

    kT = constants.k * temperature_K * constants.Avogadro / 1000.0
    conformations: List[Conformation] = []

    for macro_id, states in enumerate(macrostate_sets):
        states = np.asarray(states, dtype=int)
        macro_population = float(np.sum(pi[states]))
        free_energy = -kT * np.log(max(macro_population, 1e-12))
        committor_values = tpt_result.forward_committor[states]
        committor = float(np.mean(committor_values))
        flux = float(np.sum(flux_by_state[states]))

        kis_score = None
        if kis_result is not None:
            kis_score = float(np.mean(kis_result.kis_scores[states]))

        representative_state = int(states[np.argmax(pi[states])])

        metadata = {
            "role": _macrostate_role(macro_id, macrostate_roles),
            "macrostate_members": states.tolist(),
            "macrostate_population": macro_population,
        }

        conformations.append(
            Conformation(
                conformation_type="metastable",
                state_id=representative_state,
                macrostate_id=int(macro_id),
                frame_index=-1,
                population=macro_population,
                free_energy=free_energy,
                committor=committor,
                kis_score=kis_score,
                flux=flux,
                metadata=metadata,
            )
        )

    return conformations


def _create_transition_conformations(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: np.ndarray,
    macrostate_labels: np.ndarray,
    macrostate_memberships: np.ndarray,
    macrostate_roles: Dict[str, Set[int]],
    tse_state_ids: Set[int],
) -> List[Conformation]:
    """Create microstate-level conformations for transition regions."""

    kT = constants.k * temperature_K * constants.Avogadro / 1000.0
    source_states = set(int(s) for s in np.asarray(tpt_result.source_states))
    sink_states = set(int(s) for s in np.asarray(tpt_result.sink_states))

    conformations: List[Conformation] = []

    for state_id in range(len(pi)):
        if state_id in source_states or state_id in sink_states:
            continue

        population = float(pi[state_id])
        free_energy = -kT * np.log(max(population, 1e-12))
        committor = float(tpt_result.forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_result is not None:
            kis_score = float(kis_result.kis_scores[state_id])

        macro_id = int(macrostate_labels[state_id])
        metadata: Dict[str, Any] = {
            "macrostate_role": _macrostate_role(macro_id, macrostate_roles),
            "pcca_membership": macrostate_memberships[state_id].tolist(),
        }

        conformation_type = "tse" if state_id in tse_state_ids else "transition"
        if conformation_type == "tse":
            metadata["committor_deviation"] = abs(committor - 0.5)

        conformations.append(
            Conformation(
                conformation_type=conformation_type,
                state_id=int(state_id),
                macrostate_id=macro_id,
                frame_index=-1,
                population=population,
                free_energy=free_energy,
                committor=committor,
                kis_score=kis_score,
                flux=flux,
                metadata=metadata,
            )
        )

    return conformations


def _macrostate_role(macrostate_id: int, macrostate_roles: Dict[str, Set[int]]) -> str:
    """Return human-readable role for a macrostate."""

    if macrostate_id in macrostate_roles.get("source", set()):
        return "source"
    if macrostate_id in macrostate_roles.get("sink", set()):
        return "sink"
    return "intermediate"


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
    *,
    trajectory_locator: TrajectoryFrameLocator | None = None,
    topology_path: str | Path | None = None,
) -> None:
    """Extract and save representative structures."""
    picker = RepresentativePicker()

    preferred_order = ["metastable", "transition", "tse", "pathway"]
    all_types = {c.conformation_type for c in conformations}
    ordered_types = [t for t in preferred_order if t in all_types]
    ordered_types.extend(sorted(all_types - set(preferred_order)))

    # Group by conformation type
    for conf_type in ordered_types:
        type_conformations = [
            c for c in conformations if c.conformation_type == conf_type
        ]

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
            representatives,
            trajectories,
            type_dir,
            prefix=conf_type,
            topology_path=topology_path,
            trajectory_locator=trajectory_locator,
        )

        # Update conformations with paths
        for i, conf in enumerate(type_conformations):
            if i < len(saved_paths):
                conf.structure_path = saved_paths[i]
