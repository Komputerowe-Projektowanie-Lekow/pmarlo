import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pmarlo.data.shard import read_shard
from pmarlo.utils.path_utils import ensure_directory

from .types import ShardRequest, ShardResult, SimulationResult
from .utils import (
    _coerce_path_list,
    _timestamp,
    emit_shards_rg_rmsd_windowed,
    infer_run_cv_flag,
    infer_shard_cv_flag,
)

logger = logging.getLogger(__name__)


class ShardsMixin:
    """Methods for processing trajectory shards.

    This class is mixed into the Backend class to provide shard demultiplexing,
    discovery, and management operations.
    """

    def emit_shards(
        self,
        simulation: SimulationResult,
        request: ShardRequest,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ShardResult:
        """Generate shards from a simulation result.

        Parameters
        ----------
        simulation : SimulationResult
            The simulation result containing trajectories to shard
        request : ShardRequest
            Configuration for shard generation
        provenance : Optional[Dict[str, Any]]
            Additional metadata to include in shard provenance

        Returns
        -------
        ShardResult
            Information about the generated shards
        """
        shard_dir = self.layout.shards_dir / simulation.run_id
        ensure_directory(shard_dir)
        created = _timestamp()

        note = {
            "created_at": created,
            "kind": "demux",
            "run_id": simulation.run_id,
            "analysis_temperatures": simulation.analysis_temperatures,
            "topology": str(simulation.pdb_path),
            "traj_files": [str(p) for p in simulation.traj_files],
        }
        if provenance:
            note.update(provenance)

        # Load feature profile to determine extraction method
        from .feature_profiles import load_feature_profile, get_feature_profile_info

        profile_name = request.feature_profile
        logger.info(f"Using feature profile: {profile_name}")

        profile_info = get_feature_profile_info(profile_name)
        note["feature_profile"] = profile_name
        note["feature_type"] = profile_info.get("feature_type", "unknown")
        note["cv_biasing_compatible"] = profile_info.get("cv_biasing_compatible", False)

        # Extract shards based on profile type
        if profile_info.get("feature_type") == "cv":
            # Use traditional CV-based extraction
            logger.info("Extracting CV-based shards (Rg, RMSD)")
            shard_paths = emit_shards_rg_rmsd_windowed(
                pdb_file=simulation.pdb_path,
                traj_files=[str(p) for p in simulation.traj_files],
                out_dir=str(shard_dir),
                reference=str(request.reference) if request.reference else None,
                stride=int(max(1, request.stride)),
                temperature=float(request.temperature),
                seed_start=int(max(0, request.seed_start)),
                frames_per_shard=int(max(1, request.frames_per_shard)),
                hop_frames=(
                    int(request.hop_frames)
                    if request.hop_frames is not None and request.hop_frames > 0
                    else None
                ),
                provenance=note,
            )
        elif profile_info.get("feature_type") == "molecular":
            # Use molecular feature extraction
            logger.info("Extracting molecular feature-based shards")
            from .shard_extraction import extract_shards_with_features

            # Load feature profile with actual features
            try:
                profile = load_feature_profile(
                    profile_name,
                    spec_path=self.layout.app_root / "app" / "feature_spec.yaml" if profile_name == "molecular_custom" else None
                )
                logger.info(f"Loaded profile with {len(profile.features)} features")
            except Exception as e:
                logger.error(f"Failed to load feature profile: {e}")
                raise RuntimeError(
                    f"Could not load feature profile '{profile_name}': {e}"
                ) from e

            shard_paths = extract_shards_with_features(
                pdb_file=simulation.pdb_path,
                traj_files=[str(p) for p in simulation.traj_files],
                out_dir=str(shard_dir),
                feature_specs=profile.features,
                stride=int(max(1, request.stride)),
                temperature=float(request.temperature),
                seed_start=int(max(0, request.seed_start)),
                frames_per_shard=int(max(1, request.frames_per_shard)),
                hop_frames=(
                    int(request.hop_frames)
                    if request.hop_frames is not None and request.hop_frames > 0
                    else None
                ),
                provenance=note,
            )
        else:
            raise ValueError(f"Unknown feature type: {profile_info.get('feature_type')}")

        shard_paths = _coerce_path_list(shard_paths)

        logger.info(f"Shard emission returned {len(shard_paths)} paths")
        for idx, p in enumerate(shard_paths):
            logger.debug(f"  Shard {idx}: {p}, exists={p.exists()}, size={p.stat().st_size if p.exists() else 'N/A'}")

        # Count total frames across all shards
        n_frames = 0
        failed_reads = []
        for path in shard_paths:
            try:
                # Read the shard to count frames
                meta, data, src = read_shard(path)
                frames_in_shard = int(getattr(meta, "n_frames", 0))
                n_frames += frames_in_shard
                logger.debug(f"Shard {path.name}: {frames_in_shard} frames, data shape: {data.shape if data is not None else 'None'}")
            except Exception as e:
                logger.error(f"Failed to read shard {path}: {type(e).__name__}: {e}", exc_info=True)
                failed_reads.append(path.name)
                # Check if file exists
                if not path.exists():
                    logger.error(f"  → Shard file does not exist: {path}")
                else:
                    try:
                        size = path.stat().st_size
                        logger.error(f"  → Shard file exists, size: {size} bytes")
                        # Check for companion npz file
                        npz_path = path.with_suffix(".npz")
                        if npz_path.exists():
                            npz_size = npz_path.stat().st_size
                            logger.error(f"  → NPZ file exists, size: {npz_size} bytes")
                        else:
                            logger.error(f"  → NPZ file missing: {npz_path}")
                    except Exception as stat_err:
                        logger.error(f"  → Could not stat file: {stat_err}")
                continue

        if n_frames == 0 and len(shard_paths) > 0:
            logger.warning(
                f"Shard emission produced {len(shard_paths)} shard files but counted 0 frames total. "
                f"Failed to read {len(failed_reads)} shards: {failed_reads}. "
                f"This may indicate an issue with shard generation or file I/O."
            )

        result = ShardResult(
            run_id=simulation.run_id,
            shard_dir=shard_dir.resolve(),
            shard_paths=shard_paths,
            n_frames=int(n_frames),
            n_shards=len(shard_paths),
            temperature=float(request.temperature),
            stride=int(max(1, request.stride)),
            frames_per_shard=int(max(1, request.frames_per_shard)),
            hop_frames=(
                int(request.hop_frames)
                if request.hop_frames is not None and request.hop_frames > 0
                else None
            ),
            created_at=created,
            feature_profile=request.feature_profile,
        )

        self.state.append_shards(
            {
                "run_id": simulation.run_id,
                "directory": str(result.shard_dir),
                "paths": [str(p) for p in shard_paths],
                "temperature": float(request.temperature),
                "stride": int(max(1, request.stride)),
                "n_shards": len(shard_paths),
                "n_frames": int(n_frames),
                "frames_per_shard": int(max(1, request.frames_per_shard)),
                "hop_frames": (
                    int(request.hop_frames)
                    if request.hop_frames is not None and request.hop_frames > 0
                    else None
                ),
                "created_at": created,
                "cv_informed": provenance.get("cv_informed", False) if provenance else False,
                "cv_model_bundle": provenance.get("cv_model_bundle") if provenance else None,
                "feature_profile": request.feature_profile,
                "feature_type": profile_info.get("feature_type", "unknown"),
                "cv_biasing_compatible": profile_info.get("cv_biasing_compatible", False),
            }
        )

        logger.info(f"Generated {len(shard_paths)} shards for run {simulation.run_id}, total frames: {n_frames}")
        return result

    def _total_frames_for_paths(self, shard_json_paths: Sequence[str]) -> int:
        """Sum n_frames across the provided shard metadata JSON files."""
        total_frames = 0
        for raw_path in shard_json_paths:
            path = Path(raw_path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Shard metadata file is missing: {path}"
                ) from exc
            except Exception as exc:
                raise ValueError(
                    f"Failed to read shard metadata at {path}: {exc}"
                ) from exc

            frames_value = payload.get("n_frames")
            if frames_value is None:
                raise ValueError(
                    f"Shard metadata {path} is missing required 'n_frames'"
                )
            try:
                total_frames += int(frames_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Shard metadata {path} has non-integer 'n_frames': {frames_value!r}"
                ) from exc
        return total_frames

    def delete_shard_batch(self, index: int) -> bool:
        """Delete a shard batch and its associated files.

        Parameters
        ----------
        index : int
            Index of the shard batch in state

        Returns
        -------
        bool
            True if deletion was successful, False otherwise
        """
        entry = self.state.remove_shards(index)
        if entry is None:
            logger.warning(f"No shard batch found at index {index}")
            return False

        try:
            # Delete individual shard files
            paths = entry.get("paths", [])
            for path_str in paths:
                path = self._path_from_value(path_str)
                if path is not None and path.exists():
                    path.unlink()  # Delete the .json file
                    # Also delete associated .npz file
                    npz_path = path.with_suffix(".npz")
                    if npz_path.exists():
                        npz_path.unlink()

            # Delete shard directory if empty
            directory = self._path_from_value(entry.get("directory"))
            if directory is not None and directory.exists() and directory.is_dir():
                try:
                    directory.rmdir()  # Only removes if empty
                except OSError:
                    pass  # Directory not empty, that's OK

            logger.info(f"Deleted shard batch at index {index}")
            return True
        except Exception as e:
            logger.error(f"Error deleting shard batch {index}: {e}")
            return False

    def discover_shards(self) -> List[Path]:
        """Discover all shard JSON files in the shards directory.

        Returns
        -------
        List[Path]
            List of paths to shard JSON files
        """
        if not self.layout.shards_dir.exists():
            return []
        return sorted(self.layout.shards_dir.rglob("*.json"))

    def shard_summaries(self) -> List[Dict[str, Any]]:
        """Return shard summaries grouped by run identifier.

        Multiple shard batches can belong to the same ``run_id`` (for example,
        when demultiplexing is performed more than once). Streamlit widgets
        require stable, unique keys, so we collapse duplicates here instead of
        letting the UI render ambiguous rows.
        """

        def _merge_paths(existing: List[str], incoming: List[str]) -> List[str]:
            merged: List[str] = []
            seen: set[str] = set()
            for path in (*existing, *incoming):
                if path in seen:
                    continue
                merged.append(path)
                seen.add(path)
            return merged

        def _latest_timestamp(a: Optional[str], b: Optional[str]) -> Optional[str]:
            candidates = [ts for ts in (a, b) if isinstance(ts, str) and ts]
            if not candidates:
                return a or b
            return max(candidates)

        def _merge_analysis_temps(
            first: Optional[Sequence[Any]],
            second: Optional[Sequence[Any]],
        ) -> Optional[List[Any]]:
            values: List[Any] = []

            def _coerce_items(payload: Optional[Sequence[Any]]) -> List[Any]:
                if payload is None:
                    return []
                if isinstance(payload, (list, tuple)):
                    return list(payload)
                return [payload]

            for group in (_coerce_items(first), _coerce_items(second)):
                for item in group:
                    if item not in values:
                        values.append(item)
            return values or None

        def _coerce_temperature(value: Optional[Any]) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid temperature value for shard summary: {value!r}") from exc

        run_lookup: Dict[str, Dict[str, Any]] = {
            str(run.get("run_id", "")): run for run in self.state.runs
        }
        grouped: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        for entry in self.state.shards:
            resolved_paths: List[str] = []
            for raw_path in entry.get("paths", []):
                candidate = self._path_from_value(raw_path)
                if candidate is not None and candidate.exists():
                    resolved_paths.append(str(candidate))

            if not resolved_paths:
                continue

            run_id = str(entry.get("run_id", ""))
            if not run_id:
                raise ValueError("Shard entry is missing a run_id")
            run_entry = run_lookup.get(run_id)

            total_frames = self._total_frames_for_paths(resolved_paths)
            stored_frames = entry.get("n_frames")
            if stored_frames is not None:
                try:
                    stored_frames_int = int(stored_frames)
                except (TypeError, ValueError):
                    stored_frames_int = None
                if stored_frames_int is not None and stored_frames_int != total_frames:
                    logger.warning(
                        "Shard frame count mismatch for run %s: state=%s, actual=%s. "
                        "Using actual value.",
                        run_id,
                        stored_frames_int,
                        total_frames,
                    )

            normalized = dict(entry)
            normalized["paths"] = resolved_paths
            normalized["n_shards"] = len(resolved_paths)
            normalized["n_frames"] = total_frames
            normalized["created_at"] = entry.get("created_at") or entry.get("created")
            normalized["analysis_temperatures"] = entry.get("analysis_temperatures")
            normalized["temperature_K"] = (
                _coerce_temperature(entry.get("temperature_K"))
                or _coerce_temperature(entry.get("temperature"))
            )
            if "cv_informed" not in normalized or not normalized["cv_informed"]:
                normalized["cv_informed"] = infer_shard_cv_flag(normalized, run_entry)
            if "cv_model_bundle" not in normalized and run_entry is not None:
                model_bundle = run_entry.get("cv_model_bundle")
                if model_bundle:
                    normalized["cv_model_bundle"] = model_bundle

            existing = grouped.get(run_id)
            if existing is None:
                grouped[run_id] = normalized
                order.append(run_id)
                continue

            existing["paths"] = _merge_paths(existing["paths"], normalized["paths"])
            existing["n_shards"] = len(existing["paths"])
            existing["n_frames"] = int(existing.get("n_frames", 0)) + total_frames
            existing["cv_informed"] = (
                bool(existing.get("cv_informed")) or bool(normalized.get("cv_informed"))
            )
            if not existing.get("cv_model_bundle") and normalized.get("cv_model_bundle"):
                existing["cv_model_bundle"] = normalized["cv_model_bundle"]
            existing["analysis_temperatures"] = _merge_analysis_temps(
                existing.get("analysis_temperatures"),
                normalized.get("analysis_temperatures"),
            )
            existing["created_at"] = _latest_timestamp(
                existing.get("created_at"),
                normalized.get("created_at"),
            )

            prior_temp = existing.get("temperature_K")
            incoming_temp = normalized.get("temperature_K")
            if prior_temp is None:
                existing["temperature_K"] = incoming_temp
            elif incoming_temp is not None and abs(prior_temp - incoming_temp) > 1e-6:
                raise ValueError(
                    f"Inconsistent temperatures recorded for run '{run_id}': "
                    f"{prior_temp} vs {incoming_temp}"
                )

        info: List[Dict[str, Any]] = []
        for run_id in order:
            info.append(grouped[run_id])

        return info
