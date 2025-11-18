"""High-level API for finding conformations using TPT analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCCA+ memberships and canonical macrostate labels."""
    try:
        n_metastable = int(n_metastable)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_metastable must be an integer") from exc
    if n_metastable < 1:
        raise ValueError("n_metastable must be at least 1")
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("Transition matrix must be square to compute macrostates")
    if n_metastable > T.shape[0]:
        raise ValueError(
            f"Requested {n_metastable} macrostates but only {T.shape[0]} microstates available"
        )

    try:
        from deeptime.markov import pcca
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "PCCA+ membership computation requires the 'deeptime' package"
        ) from exc

    model = pcca(np.asarray(T, dtype=float), n_metastable)
    memberships = np.asarray(model.memberships, dtype=float)
    if memberships.ndim != 2 or memberships.shape[0] != T.shape[0]:
        raise ValueError("PCCA+ returned memberships with unexpected shape")

    # Canonicalize macrostate order by stationary population to ensure consistent labeling.
    pi_vec = np.asarray(pi, dtype=float).reshape(-1)
    if pi_vec.shape[0] != T.shape[0]:
        raise ValueError(
            "Stationary distribution size must match the transition matrix"
        )

    macro_weights = np.dot(pi_vec, memberships)
    order = np.argsort(-macro_weights)
    memberships = memberships[:, order]
    remap = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(order)}

    raw_labels = np.argmax(np.asarray(model.memberships, dtype=float), axis=1)
    macrostate_labels = np.asarray([remap[int(lbl)] for lbl in raw_labels], dtype=int)

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
    find_pathway_intermediates: bool = True,
    compute_kis: bool = True,
    uncertainty_analysis: bool = False,
    n_bootstrap: int = 100,
    representative_selection: str = "medoid",
    output_dir: Optional[str] = None,
    save_structures: bool = False,
    tse_tolerance: float = 0.05,
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
        tse_tolerance: Committor tolerance from 0.5 used to classify transition state ensemble members
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

    if uncertainty_analysis and "dtrajs" not in msm_data:
        raise ValueError(
            "Uncertainty analysis requires 'dtrajs' to be present in msm_data"
        )

    if save_structures and output_dir is None:
        raise ValueError("output_dir must be provided when save_structures=True")

    T = np.asarray(msm_data["T"])
    pi = np.asarray(msm_data["pi"])
    dtrajs = msm_data.get("dtrajs")
    features = msm_data.get("features")
    fes = msm_data.get("fes")
    its = msm_data.get("its")
    topology_path = kwargs.get("topology_path")
    trajectory_locator = kwargs.get("trajectory_locator")

    n_states = T.shape[0]
    n_metastable = _resolve_n_metastable(kwargs.get("n_metastable"), n_states)
    temperature_K = kwargs.get("temperature", 300.0)

    # Initialize result containers
    conformations: List[Conformation] = []
    tpt_result: Optional[TPTResult] = None
    kis_result: Optional[KISResult] = None
    uncertainty_results: List[UncertaintyResult] = []
    macrostate_labels: Optional[np.ndarray] = None
    macrostate_memberships: Optional[np.ndarray] = None

    try:
        macrostate_memberships, macrostate_labels = _compute_macrostate_memberships(
            T, pi, n_metastable
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to compute PCCA+ macrostates required for conformations analysis"
        ) from exc

    # Step 1: Detect or validate source/sink states
    used_auto_detection = False
    has_source = source_states is not None
    has_sink = sink_states is not None

    if has_source and has_sink:
        source_states = np.asarray(source_states)
        sink_states = np.asarray(sink_states)
        logger.info("Using provided source and sink states")
    else:
        if not auto_detect:
            missing = []
            if not has_source:
                missing.append("source_states")
            if not has_sink:
                missing.append("sink_states")
            raise ValueError(
                "Auto-detection disabled but missing "
                + " and ".join(missing)
                + "; provide both explicitly"
            )

        logger.info("Auto-detecting source and sink states")
        detector = StateDetector()
        source_states, sink_states = detector.auto_detect(
            T=T,
            pi=pi,
            fes=fes,
            its=its,
            n_states=n_metastable,
            method=auto_detect_method,
        )
        used_auto_detection = True
        logger.info(
            f"Detected {len(source_states)} source states and {len(sink_states)} sink states"
        )

    source_states = np.asarray(source_states)
    sink_states = np.asarray(sink_states)

    # Step 2: Run TPT analysis
    logger.info("Running Transition Path Theory analysis")
    tpt = TPTAnalysis(T, pi)
    tpt_result = tpt.analyze(
        source_states, sink_states, n_paths=kwargs.get("n_paths", 5)
    )

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
                macrostate_labels,
                macrostate_memberships,
            )
            conformations.extend(metastable_conformations)

        if find_transition_states or find_pathway_intermediates:
            logger.info(
                "Classifying reactive (transition) states "
                f"(tolerance={tse_tolerance})"
            )
            transition_conformations = _find_transition_states(
                tpt_result,
                pi,
                temperature_K,
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
    tse_count = sum(1 for c in conformations if c.conformation_type == "tse")
    pathway_count = sum(1 for c in conformations if c.conformation_type == "pathway")

    metadata = {
        "n_conformations": len(conformations),
        "n_pathway_intermediates": pathway_count,
        "auto_detected": used_auto_detection,
        "temperature_K": temperature_K,
        "uncertainty_analysis": uncertainty_analysis,
        "n_metastable_states": n_metastable,
        "n_transition_state_ensemble": tse_count,
    }
    if macrostate_memberships is not None:
        metadata["macrostate_memberships"] = macrostate_memberships.tolist()

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
        f"{tse_count} transition state ensemble, "
        f"{pathway_count} pathway)"
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
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")

    tolerance = float(tse_tolerance)
    if tolerance < 0 or tolerance > 0.5:
        raise ValueError("tse_tolerance must be between 0 and 0.5 inclusive")

    pi = np.asarray(pi, dtype=float).reshape(-1)
    if np.any(pi < 0):
        raise ValueError("Stationary distribution entries must be non-negative")
    n_states = pi.shape[0]

    forward_committor = np.asarray(
        tpt_result.forward_committor, dtype=float
    ).reshape(-1)
    if forward_committor.shape[0] != n_states:
        raise ValueError("forward_committor must have the same length as pi")

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)
    flux_by_state = np.asarray(flux_by_state, dtype=float).reshape(-1)
    if flux_by_state.shape[0] != n_states:
        raise ValueError("flux_by_state must have the same length as pi")

    if macrostate_labels is not None:
        macrostate_labels = np.asarray(macrostate_labels)

    kis_scores = None
    if kis_result is not None:
        kis_scores = np.asarray(kis_result.kis_scores, dtype=float).reshape(-1)
        if kis_scores.shape[0] != n_states:
            raise ValueError("kis_scores must have the same length as pi")

    source_states = set(int(s) for s in np.asarray(tpt_result.source_states, dtype=int))
    sink_states = set(int(s) for s in np.asarray(tpt_result.sink_states, dtype=int))

    for label, indices in (("Source", source_states), ("Sink", sink_states)):
        for state in indices:
            if state < 0 or state >= n_states:
                raise ValueError(
                    f"{label} state {state} is out of range for {n_states} states"
                )

    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    conformations = []
    for state_id in range(n_states):
        if state_id in source_states or state_id in sink_states:
            continue

        population = float(pi[state_id])
        free_energy = -kT * np.log(max(population, 1e-10))
        committor = float(forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_scores is not None:
            kis_score = float(kis_scores[state_id])

        macrostate_id = None
        if macrostate_labels is not None and state_id < macrostate_labels.shape[0]:
            macrostate_id = int(macrostate_labels[state_id])

        is_tse = abs(committor - 0.5) <= tolerance
        conf = Conformation(
            conformation_type="tse" if is_tse else "transition",
            state_id=int(state_id),
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            committor=committor,
            kis_score=kis_score,
            flux=flux,
            macrostate_id=macrostate_id,
        )

        conformations.append(conf)

    return conformations


def _find_metastable_states(
    tpt_result: TPTResult,
    pi: np.ndarray,
    temperature_K: float,
    kis_result: Optional[KISResult],
    flux_by_state: Optional[np.ndarray] = None,
    macrostate_labels: Optional[np.ndarray] = None,
    macrostate_memberships: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[Conformation]]:
    """Identify metastable states as source and sink sets from TPT results."""
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")

    pi = np.asarray(pi, dtype=float).reshape(-1)
    forward_committor = np.asarray(
        tpt_result.forward_committor, dtype=float
    ).reshape(-1)

    if flux_by_state is None:
        flux_by_state = _calculate_state_flux(tpt_result.flux_matrix)
    flux_by_state = np.asarray(flux_by_state, dtype=float).reshape(-1)

    if macrostate_labels is not None:
        macrostate_labels = np.asarray(macrostate_labels)

    if macrostate_memberships is not None:
        macrostate_memberships = np.asarray(macrostate_memberships, dtype=float)

    kis_scores = None
    if kis_result is not None:
        kis_scores = np.asarray(kis_result.kis_scores, dtype=float).reshape(-1)

    source_states = [int(s) for s in np.asarray(tpt_result.source_states, dtype=int)]
    sink_states = [int(s) for s in np.asarray(tpt_result.sink_states, dtype=int)]
    source_set = set(source_states)
    sink_set = set(sink_states)
    metastable_states = sorted(source_set.union(sink_set))

    if not metastable_states:
        return macrostate_labels, []

    for state_id in metastable_states:
        if state_id < 0:
            raise ValueError(f"Metastable state {state_id} is invalid")
        if state_id >= pi.shape[0]:
            raise ValueError(
                f"Metastable state {state_id} exceeds stationary distribution size"
            )
        if state_id >= forward_committor.shape[0]:
            raise ValueError(
                f"Metastable state {state_id} exceeds committor vector size"
            )
        if state_id >= flux_by_state.shape[0]:
            raise ValueError(
                f"flux_by_state does not cover metastable state {state_id}"
            )
        if kis_scores is not None and state_id >= kis_scores.shape[0]:
            raise ValueError(
                f"kis_scores do not cover metastable state {state_id}"
            )

    kT = constants.k * temperature_K * constants.Avogadro / 1000.0

    conformations: List[Conformation] = []

    for state_id in metastable_states:
        population = float(pi[state_id])
        free_energy = -kT * np.log(max(population, 1e-10))
        committor = float(forward_committor[state_id])
        flux = float(flux_by_state[state_id])

        kis_score = None
        if kis_scores is not None:
            kis_score = float(kis_scores[state_id])

        is_source = state_id in source_set
        is_sink = state_id in sink_set
        if is_source and is_sink:
            role = "source_sink"
        elif is_source:
            role = "source"
        elif is_sink:
            role = "sink"
        else:  # pragma: no cover - union guarantees membership
            raise RuntimeError(f"Metastable state {state_id} has no role assignment")

        macrostate_id = None
        if macrostate_labels is not None and state_id < macrostate_labels.shape[0]:
            macrostate_id = int(macrostate_labels[state_id])

        macro_members = None
        if (
            macrostate_memberships is not None
            and state_id < macrostate_memberships.shape[0]
        ):
            macro_members = macrostate_memberships[state_id].tolist()

        conf_metadata = {"role": role}
        if macro_members is not None:
            conf_metadata["macrostate_members"] = macro_members

        conf = Conformation(
            conformation_type="metastable",
            state_id=int(state_id),
            frame_index=-1,
            population=population,
            free_energy=free_energy,
            committor=committor,
            kis_score=kis_score,
            flux=flux,
            macrostate_id=macrostate_id,
            metadata=conf_metadata,
        )

        conformations.append(conf)

    return macrostate_labels, conformations


def _calculate_state_flux(flux_matrix: np.ndarray) -> np.ndarray:
    """Compute the mean reactive flux through each state.

    For each state we average the magnitudes of every incident edge (incoming
    plus outgoing). Using a mean instead of a sum prevents hubs with many weak
    connections from appearing dominant while ensuring the score is shared by
    both endpoints of each edge.
    """
    flux_matrix = np.asarray(flux_matrix, dtype=np.float64)
    outgoing_sum = np.sum(flux_matrix, axis=1)
    outgoing_edges = np.count_nonzero(flux_matrix, axis=1)

    incoming_sum = np.sum(flux_matrix, axis=0)
    incoming_edges = np.count_nonzero(flux_matrix, axis=0)

    total_sum = outgoing_sum + incoming_sum
    total_edges = outgoing_edges + incoming_edges

    return np.divide(
        total_sum,
        np.maximum(total_edges, 1),
        out=np.zeros_like(total_sum),
        where=total_edges > 0,
    )


def _update_with_representatives(
    conformations: List[Conformation],
    representatives: List[Tuple[int, int, int, int]],
) -> None:
    """Update conformations with representative frame information.

    For each Conformation whose state_id appears in `representatives`,
    sets:
        - frame_index
        - trajectory_index
        - local_frame_index

    Conformations without a representative are left unchanged.

    Raises:
        ValueError:
            - if there are duplicate state_ids in `representatives`
            - if a representative is provided for a state_id that does not
              exist in `conformations`
    """

    # Enforce: at most one representative per state_id
    state_ids = [state_id for state_id, _, _, _ in representatives]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("Duplicate state_id entries in representatives")

    # Map: state_id -> (frame_idx, traj_idx, local_idx)
    rep_dict = {
        state_id: (frame_idx, traj_idx, local_idx)
        for state_id, frame_idx, traj_idx, local_idx in representatives
    }

    # Optional but very useful: detect representatives for non-existing states
    conf_state_ids = {c.state_id for c in conformations}
    missing_states = set(rep_dict.keys()) - conf_state_ids
    if missing_states:
        raise ValueError(
            f"Representative(s) provided for unknown state_id(s): {sorted(missing_states)}"
        )

    # Update conformations in place
    for conf in conformations:
        rep = rep_dict.get(conf.state_id)
        if rep is None:
            # No representative for this state, leave as is
            continue

        frame_idx, traj_idx, local_idx = rep
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

    # Process each conformation type separately
    for conf_type in ["metastable", "transition", "pathway"]:
        # All conformations of this type
        type_conformations = [
            c for c in conformations if c.conformation_type == conf_type
        ]

        if not type_conformations:
            continue

        # Representatives and the corresponding Conformation objects
        representatives: List[Tuple[int, int, int, int]] = []
        representative_confs: List[Conformation] = []

        for c in type_conformations:
            # Negative frame_index means "no representative frame" for this conformation
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
            representative_confs.append(c)

        # Nothing to extract for this type
        if not representatives:
            continue

        # Save structures for this type
        type_dir = str(Path(output_dir) / conf_type)
        saved_paths = picker.extract_structures(
            representatives,
            trajectories,
            type_dir,
            prefix=conf_type,
            topology_path=topology_path,
            trajectory_locator=trajectory_locator,
        )

        if len(saved_paths) != len(representative_confs):
            raise RuntimeError(
                "Number of saved paths does not match number of representative "
                f"conformations for type '{conf_type}': "
                f"{len(saved_paths)} vs {len(representative_confs)}"
            )

        # Assign paths only to conformations that actually had representatives
        for conf, path in zip(representative_confs, saved_paths):
            conf.structure_path = path

