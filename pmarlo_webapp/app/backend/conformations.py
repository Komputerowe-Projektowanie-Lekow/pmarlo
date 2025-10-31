from pathlib import Path
from typing import Sequence
from .types import ConformationsConfig, ConformationsResult
from pmarlo.conformations.representative_picker import (
    TrajectoryFrameLocator,
    TrajectorySegment,
)

def load_conformations(
        self, handle: int | Mapping[str, Any]
) -> Optional[ConformationsResult]:
    if isinstance(handle, Mapping):
        entry = dict(handle)
        state_idx = entry.get("state_index")
        try:
            idx = int(state_idx) if state_idx is not None else None
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(self.state.conformations):
            entry = dict(self.state.conformations[idx])
            entry["state_index"] = idx
            return self._load_conformations_from_entry(entry)
        return self._load_conformations_from_entry(entry)

    index = int(handle)
    if index < 0 or index >= len(self.state.conformations):
        return None
    entry = dict(self.state.conformations[index])
    entry["state_index"] = index
    return self._load_conformations_from_entry(entry)

def _reconcile_conformation_state(self) -> None:
    """Drop conformations entries whose artifacts no longer exist."""

    try:
        to_delete: List[int] = []
        for i, entry in enumerate(list(self.state.conformations)):
            output_dir = self._path_from_value(
                entry.get("output_dir") or entry.get("directory")
            )
            summary_path = self._path_from_value(
                entry.get("summary") or entry.get("summary_path")
            )

            exists = False
            if summary_path is not None and summary_path.exists():
                exists = True
            elif output_dir is not None and output_dir.exists():
                exists = True

            if not exists:
                to_delete.append(i)

        for idx in reversed(to_delete):
            try:
                self.state.remove_conformations(idx)
            except Exception:
                pass
    except Exception:
        # Non-fatal cleanup failure
        pass

def _load_conformations_from_entry(
        self, entry: Mapping[str, Any]
) -> Optional[ConformationsResult]:
    output_dir = self._path_from_value(
        entry.get("output_dir") or entry.get("directory")
    )
    if output_dir is None:
        return None

    summary_raw = entry.get("summary") or entry.get("summary_path")
    summary_path = self._path_from_value(summary_raw)
    if summary_path is None:
        summary_path = output_dir / "conformations_summary.json"

    if not summary_path.exists():
        alt_summary = output_dir / "conformations_summary.json"
        if alt_summary.exists():
            summary_path = alt_summary
        else:
            return None

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    else:
        _rewrite_strings_in_place(payload, self.layout.normalize_path_string)

    tpt_summary = dict(payload.get("tpt") or {})
    metastable_states = dict(payload.get("metastable_states") or {})
    transition_states = list(payload.get("transition_states") or [])
    pathways = list(payload.get("pathways") or [])
    created_at = str(payload.get("created_at", entry.get("created_at", _timestamp())))

    config_payload = payload.get("config") or entry.get("config") or {}
    config_obj = self._conformations_config_from_entry(config_payload)

    plots: Dict[str, Path] = {}
    entry_plots = entry.get("plots")
    if isinstance(entry_plots, Mapping):
        for name, plot_path in entry_plots.items():
            candidate = self._path_from_value(plot_path)
            if candidate is not None and candidate.exists():
                plots[str(name)] = candidate.resolve()

    try:
        for plot_path in output_dir.glob("*.png"):
            plots.setdefault(plot_path.stem, plot_path.resolve())
    except Exception:
        pass

    try:
        representative_pdbs = [p.resolve() for p in sorted(output_dir.glob("*.pdb"))]
    except Exception:
        representative_pdbs = []

    tpt_converged = bool(
        entry.get("tpt_converged", tpt_summary.get("tpt_converged", True))
    )

    tpt_iterations = entry.get("tpt_pathway_iterations")
    if tpt_iterations is None:
        tpt_iterations = tpt_summary.get("pathway_iterations")

    tpt_max_iterations = entry.get("tpt_pathway_max_iterations")
    if tpt_max_iterations is None:
        tpt_max_iterations = tpt_summary.get("pathway_max_iterations")

    error_message = entry.get("error")
    error_str = str(error_message) if error_message is not None else None

    return ConformationsResult(
        output_dir=output_dir.resolve(),
        tpt_summary=tpt_summary,
        metastable_states=metastable_states,
        transition_states=transition_states,
        pathways=pathways,
        representative_pdbs=representative_pdbs,
        plots=plots,
        created_at=created_at,
        config=config_obj,
        error=error_str,
        tpt_converged=bool(tpt_converged),
        tpt_pathway_iterations=(
            int(tpt_iterations) if tpt_iterations is not None else None
        ),
        tpt_pathway_max_iterations=(
            int(tpt_max_iterations) if tpt_max_iterations is not None else None
        ),
    )

def _conformations_config_from_entry(
        self, payload: Any
) -> ConformationsConfig:
    if not isinstance(payload, Mapping):
        return ConformationsConfig()

    data: Dict[str, Any] = dict(payload)
    for key in (
            "topology_pdb",
            "deeptica_projection_path",
            "deeptica_metadata_path",
    ):
        value = data.get(key)
        if value:
            try:
                candidate = Path(value).expanduser()
            except Exception:
                candidate = Path(str(value)).expanduser()
            if candidate.is_absolute():
                candidate = self.layout.rebase_legacy_path(candidate)
            data[key] = candidate
        else:
            data[key] = None

    thresholds = data.get("committor_thresholds")
    if isinstance(thresholds, (list, tuple)):
        try:
            data["committor_thresholds"] = tuple(float(v) for v in thresholds)
        except Exception:
            data.pop("committor_thresholds", None)

    for state_key in ("source_states", "sink_states"):
        states = data.get(state_key)
        if isinstance(states, (list, tuple)):
            try:
                data[state_key] = [int(s) for s in states]
            except Exception:
                data[state_key] = [int(float(s)) for s in states if s is not None]

    cluster_seed = data.get("cluster_seed")
    if cluster_seed is not None:
        try:
            data["cluster_seed"] = int(cluster_seed)
        except (TypeError, ValueError):
            data["cluster_seed"] = None

    try:
        return ConformationsConfig(**data)
    except Exception:
        return ConformationsConfig()

def list_conformations(self) -> List[Dict[str, Any]]:
    """Return recorded conformations analyses with resolved paths."""

    self._reconcile_conformation_state()
    entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(self.state.conformations):
        data = dict(entry)
        data["state_index"] = idx
        output_dir_raw = data.get("output_dir") or data.get("directory")
        output_dir_path = self._path_from_value(output_dir_raw)
        if output_dir_path is not None:
            data["output_dir"] = str(output_dir_path)
        summary_raw = data.get("summary") or data.get("summary_path")
        summary_path = self._path_from_value(summary_raw)
        if summary_path is not None:
            data["summary"] = str(summary_path)
        entries.append(data)

    entries.sort(key=lambda e: str(e.get("created_at", "")), reverse=True)
    return entries

def run_conformations_analysis(
        self,
        shard_jsons: Sequence[Path],
        config: ConformationsConfig,
) -> ConformationsResult:
    """Run TPT conformations analysis on shards.

    Args:
        shard_jsons: Paths to shard JSON files
        config: Configuration for conformations analysis

    Returns:
        ConformationsResult with outputs and metadata
    """
    from pmarlo.analysis.project_cv import apply_whitening_from_metadata
    from pmarlo.conformations import find_conformations
    from pmarlo.conformations.visualizations import (
        plot_pcca_states,
        plot_tpt_summary,
    )
    from pmarlo.markov_state_model.clustering import cluster_microstates
    from pmarlo.markov_state_model.reduction import reduce_features

    stamp = _timestamp()
    output_dir = self.layout.bundles_dir / f"conformations-{stamp}"
    ensure_directory(output_dir)
    config_dict: Dict[str, Any] = {}
    summary_path = output_dir / "conformations_summary.json"

    try:
        # Load shards using the same method as MSM building
        logger.info(f"Loading {len(shard_jsons)} shards for conformations analysis")
        shards = [Path(p).resolve() for p in shard_jsons]
        if not shards:
            raise ValueError("No shards selected for conformations analysis")

        dataset = load_shards_as_dataset(shards)

        # Extract features from dataset
        if "X" not in dataset or len(dataset["X"]) == 0:
            raise ValueError("No feature data found in shards")

        features = np.asarray(dataset["X"], dtype=float)
        logger.info(f"Loaded {features.shape[0]} frames with {features.shape[1]} features")

        if config.topology_pdb is None:
            raise ValueError(
                "A topology PDB must be specified for conformations analysis."
            )

        raw_topology = Path(config.topology_pdb)
        if raw_topology.is_absolute():
            topology_pdb = raw_topology.expanduser().resolve()
        else:
            topology_pdb = (self.layout.workspace_dir / raw_topology).expanduser().resolve()

        if not topology_pdb.exists():
            raise FileNotFoundError(
                f"Topology PDB {topology_pdb} does not exist."
            )

        shard_meta_list = dataset.get("__shards__", [])
        if not shard_meta_list:
            raise ValueError(
                "Shard metadata missing from aggregated dataset; cannot locate trajectories."
            )

        locator = self._build_trajectory_locator(shards, shard_meta_list)
        logger.info(
            "Resolved %d trajectory segments for representative extraction",
            len(locator.segments),
        )

        cv_method = (config.cv_method or "tica").strip().lower()
        tica_dim = (
            int(config.tica_dim)
            if config.tica_dim is not None
            else int(config.n_components)
        )
        if cv_method == "deeptica":
            if config.deeptica_projection_path is None:
                raise ValueError(
                    "deeptica_projection_path is required when cv_method='deeptica'"
                )
            projection_path = _resolve_workspace_path(
                self.layout.workspace_dir,
                Path(config.deeptica_projection_path),
            )
            logger.info("Loading precomputed DeepTICA projection from %s", projection_path)
            features_reduced = _load_projection_matrix(projection_path)
            if features_reduced.shape[0] != features.shape[0]:
                raise ValueError(
                    "DeepTICA projection frame count does not match loaded features"
                )
            if config.deeptica_metadata_path is not None:
                metadata_path = _resolve_workspace_path(
                    self.layout.workspace_dir,
                    Path(config.deeptica_metadata_path),
                )
                logger.info("Applying DeepTICA whitening metadata from %s", metadata_path)
                metadata = _load_metadata_mapping(metadata_path)
                features_reduced, _ = apply_whitening_from_metadata(
                    np.asarray(features_reduced, dtype=float), metadata
                )
            else:
                features_reduced = np.asarray(features_reduced, dtype=float)
        elif cv_method == "tica":
            logger.info(
                "Reducing features with TICA (n_components=%d)",
                tica_dim,
            )
            features_reduced = reduce_features(
                features,
                method="tica",
                lag=config.lag,
                n_components=tica_dim,
            )
        else:
            raise ValueError(f"Unsupported cv_method '{config.cv_method}' for conformations")

        # Clustering
        cluster_mode = (config.cluster_mode or "kmeans").strip().lower()
        method_alias = {
            "kmeans": "kmeans",
            "minibatchkmeans": "minibatchkmeans",
            "auto": "auto",
        }
        if cluster_mode not in method_alias:
            raise ValueError(
                "Unsupported cluster_mode for conformations analysis: "
                f"{config.cluster_mode!r}."
            )

        cluster_kwargs: Mapping[str, Any]
        if config.kmeans_kwargs is None:
            cluster_kwargs = {}
        elif isinstance(config.kmeans_kwargs, Mapping):
            cluster_kwargs = dict(config.kmeans_kwargs)
        else:
            raise TypeError(
                "ConformationsConfig.kmeans_kwargs must be a mapping; "
                f"received {type(config.kmeans_kwargs).__name__}."
            )

        n_clusters = int(config.n_clusters)
        if n_clusters <= 0:
            raise ValueError(
                "ConformationsConfig.n_clusters must be a positive integer"
            )

        total_frames = int(features_reduced.shape[0])
        if total_frames == 0:
            raise ValueError("No frames available after dimensionality reduction")

        frames_per_cluster = total_frames / float(n_clusters)
        if frames_per_cluster > 10_000:
            logger.warning(
                (
                    "Using %d clusters for %d frames (~%.0f frames/cluster) may be too coarse. "
                    "Increase n_clusters to improve transition state resolution."
                ),
                n_clusters,
                total_frames,
                frames_per_cluster,
            )

        logger.info(
            "Clustering into %d microstates using %s (seed=%s, n_init=%d, kwargs=%s)",
            n_clusters,
            method_alias[cluster_mode],
            "None" if config.cluster_seed is None else int(config.cluster_seed),
            int(config.kmeans_n_init),
            cluster_kwargs,
        )

        clustering_result = cluster_microstates(
            features_reduced,
            method=method_alias[cluster_mode],
            n_states=n_clusters,
            random_state=(
                None
                if config.cluster_seed is None
                else int(config.cluster_seed)
            ),
            n_init=int(config.kmeans_n_init),
            **cluster_kwargs,
        )
        # Extract labels from ClusteringResult object
        if clustering_result.centers is None:
            raise ValueError(
                "Clustering did not return cluster centers required for PCCA visualization."
            )
        cluster_centers = np.asarray(clustering_result.centers, dtype=float)
        if cluster_centers.ndim != 2:
            raise ValueError(
                "Cluster centers must be a 2D array to generate the PCCA visualization."
            )
        if cluster_centers.shape[1] < 2:
            raise ValueError(
                "At least two TICA dimensions are required to plot PCCA metastable states."
            )
        tica_cluster_coords = cluster_centers[:, :2]
        labels = clustering_result.labels
        n_states = int(np.max(labels) + 1)

        # Build MSM using deeptime reversible estimator
        logger.info(f"Building MSM (lag={config.lag}) using deeptime")
        if dt is None:
            raise RuntimeError(
                "Deeptime library is required for reversible MSM estimation."
            )

        estimator = dt.markov.msm.MaximumLikelihoodMSM(
            lagtime=config.lag, reversible=True
        )
        msm_model = estimator.fit([labels]).fetch_model()
        T = msm_model.transition_matrix
        pi = msm_model.stationary_distribution

        if not _is_transition_matrix_reversible(T, pi):
            raise ValueError(
                "Transition matrix is not reversible; TPT requires detailed balance."
            )

        # Run TPT conformations analysis
        logger.info("Running TPT conformations analysis")

        # Prepare MSM data dictionary
        msm_data = {
            'T': T,
            'pi': pi,
            'dtrajs': [labels],
            'features': features_reduced,
        }

        conf_result = find_conformations(
            msm_data=msm_data,
            source_states=np.array(config.source_states) if config.source_states else None,
            sink_states=np.array(config.sink_states) if config.sink_states else None,
            auto_detect=config.auto_detect_states,
            auto_detect_method='auto',
            find_transition_states=True,
            find_metastable_states=True,
            find_pathway_intermediates=True,
            compute_kis=config.compute_kis,
            uncertainty_analysis=config.uncertainty_analysis,
            n_bootstrap=config.bootstrap_samples,
            lag=int(config.lag),
            representative_selection='medoid',
            output_dir=str(output_dir),
            save_structures=True,
            topology_path=str(topology_pdb),
            trajectory_locator=locator,
            tica__dim=tica_dim,
            committor_thresholds=tuple(config.committor_thresholds),
            n_metastable=config.n_metastable,  # Pass the n_metastable parameter
        )

        macro_memberships_data = conf_result.metadata.get("macrostate_memberships")
        if macro_memberships_data is None:
            raise ValueError(
                "Conformations analysis did not return PCCA memberships required for visualization."
            )
        pcca_memberships = np.asarray(macro_memberships_data, dtype=float)
        if pcca_memberships.ndim != 2:
            raise ValueError(
                "PCCA memberships must be a 2D array to generate the metastable state plot."
            )
        if pcca_memberships.shape[0] != tica_cluster_coords.shape[0]:
            raise ValueError(
                "The number of PCCA membership rows does not match the number of microstate clusters."
            )

        pcca_plot_path = output_dir / "pcca_states.png"
        plot_pcca_states(tica_cluster_coords, pcca_memberships, str(pcca_plot_path))

        tpt_result = conf_result.tpt_result
        if tpt_result is None:
            raise RuntimeError(
                "TPT analysis did not produce a result. Ensure the transition matrix is reversible and source/sink states are valid."
            )

        # Generate visualizations
        logger.info("Generating visualizations")
        plot_tpt_summary(tpt_result, str(output_dir))
        plots = {"pcca_states": pcca_plot_path}
        for plot_name in ("committors", "flux_network", "pathways"):
            plot_path = output_dir / f"{plot_name}.png"
            if plot_path.exists():
                plots[plot_name] = plot_path

        # Extract summary data
        tpt_summary = {
            "rate": float(tpt_result.rate),
            "mfpt": float(tpt_result.mfpt),
            "total_flux": float(tpt_result.total_flux),
            "n_pathways": len(tpt_result.pathways),
            "source_states": tpt_result.source_states.tolist(),
            "sink_states": tpt_result.sink_states.tolist(),
            "tpt_converged": bool(tpt_result.tpt_converged),
            "pathway_iterations": int(tpt_result.pathway_iterations),
            "pathway_max_iterations": int(tpt_result.pathway_max_iterations),
        }

        metastable_states: Dict[str, Dict[str, Any]] = {}
        for conf in conf_result.get_metastable_states():
            macro_id = (
                int(conf.macrostate_id)
                if conf.macrostate_id is not None
                else int(conf.state_id)
            )
            micro_ids = conf.metadata.get("microstate_ids", [])
            n_states = len(micro_ids) if isinstance(micro_ids, list) else 0
            metastable_states[str(macro_id)] = {
                "population": float(conf.population),
                "n_states": n_states,
                "representative_pdb": (
                    str(conf.structure_path)
                    if conf.structure_path is not None
                    else None
                ),
            }

        transition_states: List[Dict[str, Any]] = []
        for conf in conf_result.get_transition_states():
            transition_states.append(
                {
                    "committor": float(conf.committor) if conf.committor is not None else 0.0,
                    "state_index": int(conf.state_id),
                    "representative_pdb": (
                        str(conf.structure_path)
                        if conf.structure_path is not None
                        else None
                    ),
                }
            )

        pathways: List[List[int]] = []
        for path in tpt_result.pathways:
            pathways.append([int(state) for state in path])

        representative_pdbs = []
        for f in output_dir.glob("*.pdb"):
            representative_pdbs.append(f)

        # Save summary JSON
        config_dict = asdict(config)
        if config.topology_pdb is not None:
            config_dict["topology_pdb"] = str(
                Path(config.topology_pdb)
                if isinstance(config.topology_pdb, Path)
                else config.topology_pdb
            )
        if config.deeptica_projection_path is not None:
            config_dict["deeptica_projection_path"] = str(
                Path(config.deeptica_projection_path)
                if isinstance(config.deeptica_projection_path, Path)
                else config.deeptica_projection_path
            )
        if config.deeptica_metadata_path is not None:
            config_dict["deeptica_metadata_path"] = str(
                Path(config.deeptica_metadata_path)
                if isinstance(config.deeptica_metadata_path, Path)
                else config.deeptica_metadata_path
            )
        with open(summary_path, "w") as f:
            json.dump({
                "tpt": tpt_summary,
                "metastable_states": metastable_states,
                "transition_states": transition_states,
                "pathways": pathways,
                "config": config_dict,
                "created_at": stamp,
            }, f, indent=2)

        logger.info(f"Conformations analysis complete. Output saved to {output_dir}")

        conf_result = ConformationsResult(
            output_dir=output_dir,
            tpt_summary=tpt_summary,
            metastable_states=metastable_states,
            transition_states=transition_states,
            pathways=pathways,
            representative_pdbs=representative_pdbs,
            plots=plots,
            created_at=stamp,
            config=config,
            tpt_converged=bool(tpt_result.tpt_converged),
            tpt_pathway_iterations=int(tpt_result.pathway_iterations),
            tpt_pathway_max_iterations=int(tpt_result.pathway_max_iterations),
        )

        self.state.append_conformations(
            {
                "output_dir": str(output_dir.resolve()),
                "summary": str(summary_path.resolve()),
                "created_at": stamp,
                "tpt_summary": _sanitize_artifacts(tpt_summary),
                "metastable_states": _sanitize_artifacts(metastable_states),
                "transition_states": _sanitize_artifacts(transition_states),
                "pathways": _sanitize_artifacts(pathways),
                "plots": {name: str(path) for name, path in plots.items()},
                "config": _sanitize_artifacts(config_dict),
                "tpt_converged": bool(tpt_result.tpt_converged),
                "tpt_pathway_iterations": int(tpt_result.pathway_iterations),
                "tpt_pathway_max_iterations": int(
                    tpt_result.pathway_max_iterations
                ),
            }
        )

        return conf_result

    except Exception as e:
        logger.error(f"Conformations analysis failed: {e}", exc_info=True)
        self.state.append_conformations(
            {
                "output_dir": str(output_dir.resolve()),
                "summary": str(summary_path.resolve()),
                "created_at": stamp,
                "config": _sanitize_artifacts(config_dict),
                "error": str(e),
            }
        )
        return ConformationsResult(
            output_dir=output_dir,
            tpt_summary={},
            metastable_states={},
            transition_states=[],
            pathways=[],
            representative_pdbs=[],
            plots={},
            created_at=stamp,
            config=config,
            error=str(e),
            tpt_converged=True,
            tpt_pathway_iterations=None,
            tpt_pathway_max_iterations=None,
        )


def _build_trajectory_locator(
        self,
        shard_paths: Sequence[Path],
        shard_meta_list: Sequence[Mapping[str, Any]],
) -> TrajectoryFrameLocator:
    if len(shard_paths) != len(shard_meta_list):
        raise ValueError(
            "Shard metadata length mismatch; cannot map trajectories reliably."
        )

    segments: List[TrajectorySegment] = []
    for idx, (shard_path, shard_meta) in enumerate(zip(shard_paths, shard_meta_list)):
        if not isinstance(shard_meta, Mapping):
            raise TypeError(
                f"Shard metadata entry {idx} is not a mapping; unable to resolve trajectories."
            )

        start_raw = shard_meta.get("start")
        stop_raw = shard_meta.get("stop")
        if start_raw is None or stop_raw is None:
            raise ValueError(
                f"Shard metadata for {shard_path.name} must include start/stop offsets"
            )
        start = int(start_raw)
        stop = int(stop_raw)
        if stop <= start:
            raise ValueError(
                f"Shard {shard_path.name} reports non-positive frame span ({start}->{stop})"
            )

        frames_loaded = int(shard_meta.get("frames_loaded", stop - start))
        if frames_loaded != stop - start:
            raise ValueError(
                f"Shard {shard_path.name} has inconsistent frame counts (loaded={frames_loaded}, span={stop - start})"
            )

        source = shard_meta.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(
                f"Shard metadata for {shard_path.name} is missing provenance 'source' details"
            )

        frame_range = source.get("range") or source.get("frame_range")
        if not (isinstance(frame_range, (list, tuple)) and len(frame_range) == 2):
            raise ValueError(
                f"Shard metadata for {shard_path.name} must declare frame range for trajectory extraction"
            )
        local_start = int(frame_range[0])
        local_stop = int(frame_range[1])
        if local_stop - local_start != frames_loaded:
            raise ValueError(
                f"Shard {shard_path.name} frame range ({local_start}->{local_stop}) does not match feature count {frames_loaded}"
            )

        trajectory_names = self._extract_trajectory_names(source)
        trajectory_path = self._resolve_trajectory_path(shard_path, trajectory_names)

        segments.append(
            TrajectorySegment(
                path=trajectory_path,
                start=start,
                stop=stop,
                local_start=local_start,
            )
        )

    segments.sort(key=lambda seg: seg.start)
    for prev, current in zip(segments, segments[1:]):
        if current.start < prev.stop:
            raise ValueError(
                "Shard frame intervals overlap; cannot resolve representative frames to unique trajectories."
            )

    return TrajectoryFrameLocator(tuple(segments))

def _resolve_trajectory_path(
        self, shard_path: Path, raw_names: Sequence[str]
) -> Path:
    if not raw_names:
        raise ValueError(
            f"Shard {shard_path} does not declare any trajectory file references"
        )

    shard_dir = shard_path.parent.resolve()
    search_bases = [shard_dir, self.layout.workspace_dir.resolve()]

    for name in raw_names:
        candidate = self.layout.rebase_legacy_path(Path(name))
        stem = candidate.stem
        if candidate.is_absolute():
            targets = [candidate.resolve()]
        else:
            targets = [(base / candidate).resolve() for base in search_bases]

        for target in targets:
            resolved = self._maybe_resolve_structure_file(target, stem, shard_dir)
            if resolved is not None:
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Trajectory file {resolved} referenced by {shard_path} does not exist"
                    )
                return resolved.resolve()

    raise FileNotFoundError(
        f"Could not resolve trajectory file for shard {shard_path.name}."
    )

def _maybe_resolve_structure_file(
        self, target: Path, stem: str, shard_dir: Path
) -> Optional[Path]:
    suffix = target.suffix.lower()
    if target.exists():
        if suffix in _STRUCTURE_EXTENSIONS:
            return target.resolve()
        if suffix in {".npz", ".npy"}:
            alt = self._search_structure_by_stem(target.parent, stem)
            if alt is None:
                raise FileNotFoundError(
                    f"Feature archive {target} does not have a matching structural trajectory"
                )
            return alt
        raise ValueError(
            f"Unsupported trajectory file extension '{target.suffix}' for {target}"
        )

    alt = self._search_structure_by_stem(target.parent, stem)
    if alt is not None:
        return alt
    if target.parent != shard_dir:
        alt = self._search_structure_by_stem(shard_dir, stem)
        if alt is not None:
            return alt
    return None

def _search_structure_by_stem(self, base: Path, stem: str) -> Optional[Path]:
    if not stem:
        return None
    base = base.resolve()
    for ext in _STRUCTURE_EXTENSIONS:
        candidate = (base / f"{stem}{ext}").resolve()
        if candidate.exists():
            return candidate
    return None

def _extract_trajectory_names(self, source: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("traj_files", "trajectories"):
        entries = source.get(key)
        if isinstance(entries, (list, tuple)):
            for entry in entries:
                if isinstance(entry, str) and entry:
                    names.append(entry)
    primary = source.get("traj") or source.get("trajectory") or source.get("path")
    if isinstance(primary, str) and primary:
        names.append(primary)

    deduped: List[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped

class ConformationsMixin:
    """Methods for TPT conformations analysis."""

    def run_conformations_analysis(
            self,
            shard_jsons: Sequence[Path],
            config: ConformationsConfig,
    ) -> ConformationsResult:

    # ... your run_conformations_analysis implementation

    def load_conformations(self, handle) -> Optional[ConformationsResult]:

    # ... your load_conformations implementation

    def _build_trajectory_locator(...) -> TrajectoryFrameLocator:

    # ... your implementation

    def _reconcile_conformation_state(self) -> None:
# ... your implementation
