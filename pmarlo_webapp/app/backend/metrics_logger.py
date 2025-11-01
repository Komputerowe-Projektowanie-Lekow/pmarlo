"""Metrics logging system for MSM/FES analysis builds."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import numpy as np

from .types import BuildArtifact

logger = logging.getLogger(__name__)


def _serialize_for_json(obj: Any) -> Any:
    """Convert object to JSON-serializable format.

    Handles numpy arrays, numpy scalars, and other non-serializable types.
    """
    if isinstance(obj, np.ndarray):
        if obj.size < 100:  # Only serialize small arrays
            return obj.tolist()
        else:
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


class MetricsLogger:
    """Logger for saving analysis metrics to structured text files."""

    def __init__(self, logs_dir: Path):
        """Initialize the metrics logger.

        Parameters
        ----------
        logs_dir : Path
            Base directory for storing logs.
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_msm_fes_build(
        self,
        artifact: BuildArtifact,
        build_id: Optional[str] = None,
    ) -> Path:
        """Log all metrics from an MSM/FES build to organized files.

        Parameters
        ----------
        artifact : BuildArtifact
            The build artifact containing all metrics and results.
        build_id : Optional[str]
            Custom build ID. If None, generates from bundle name and timestamp.

        Returns
        -------
        Path
            Directory where metrics were saved.
        """
        if build_id is None:
            bundle_name = artifact.bundle_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            build_id = f"{bundle_name}_{timestamp}"

        metrics_dir = self.logs_dir / f"MSM_FES_BUILD_{build_id}"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Logging MSM/FES build metrics to: {metrics_dir}")

        # Log each category of metrics
        self._log_msm_metrics(artifact, metrics_dir)
        self._log_fes_metrics(artifact, metrics_dir)
        self._log_diagnostics(artifact, metrics_dir)
        self._log_artifacts(artifact, metrics_dir)
        self._log_build_summary(artifact, metrics_dir)

        logger.info(f"Successfully logged all metrics for build: {build_id}")
        return metrics_dir

    def _log_msm_metrics(self, artifact: BuildArtifact, metrics_dir: Path) -> None:
        """Log MSM-specific metrics."""
        msm_file = metrics_dir / "msm_metrics.txt"

        with open(msm_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MSM METRICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Bundle Path: {artifact.bundle_path}\n")
            f.write(f"Dataset Hash: {artifact.dataset_hash}\n")
            f.write(f"Created At: {artifact.created_at}\n\n")

            br = artifact.build_result
            if br and br.transition_matrix is not None:
                T = br.transition_matrix
                f.write(f"Number of States: {T.shape[0]}\n")
                f.write(f"Transition Matrix Shape: {T.shape}\n")
                f.write(f"Diagonal Mass (mean): {np.trace(T) / T.shape[0]:.6f}\n")
                f.write(f"Min Transition Probability: {np.min(T):.8f}\n")
                f.write(f"Max Transition Probability: {np.max(T):.8f}\n")
                f.write(f"Mean Transition Probability: {np.mean(T):.8f}\n")
                f.write(f"Std Transition Probability: {np.std(T):.8f}\n\n")

                if br.stationary_distribution is not None:
                    pi = br.stationary_distribution
                    f.write("Stationary Distribution:\n")
                    f.write(f"  Most Populated State: {np.argmax(pi)}\n")
                    f.write(f"  Max Population: {np.max(pi):.6f}\n")
                    f.write(f"  Min Population: {np.min(pi):.6f}\n")
                    f.write(f"  Mean Population: {np.mean(pi):.6f}\n")
                    f.write(f"  Std Population: {np.std(pi):.6f}\n\n")

                if artifact.debug_summary:
                    f.write("Build Statistics:\n")
                    for key, value in artifact.debug_summary.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

                if artifact.analysis_msm_n_states:
                    f.write(f"Analysis MSM States: {artifact.analysis_msm_n_states}\n")
                if artifact.tau_frames:
                    f.write(f"Tau Frames: {artifact.tau_frames}\n")
                if artifact.effective_tau_frames:
                    f.write(f"Effective Tau Frames: {artifact.effective_tau_frames}\n")
                if artifact.effective_stride_max:
                    f.write(f"Effective Stride Max: {artifact.effective_stride_max}\n")
            else:
                f.write("No MSM data available\n")

        logger.debug(f"Saved MSM metrics to: {msm_file}")

    def _log_fes_metrics(self, artifact: BuildArtifact, metrics_dir: Path) -> None:
        """Log FES-specific metrics."""
        fes_file = metrics_dir / "fes_metrics.txt"

        with open(fes_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FREE ENERGY SURFACE METRICS\n")
            f.write("=" * 80 + "\n\n")

            br = artifact.build_result
            if br and br.fes is not None:
                fes_obj = br.fes

                if hasattr(fes_obj, "F"):
                    F = fes_obj.F
                    f.write(f"FES Grid Shape: {F.shape}\n")
                    f.write(f"Min Free Energy: {np.nanmin(F):.4f} kJ/mol\n")
                    f.write(f"Max Free Energy: {np.nanmax(F):.4f} kJ/mol\n")
                    f.write(f"Mean Free Energy: {np.nanmean(F):.4f} kJ/mol\n")
                    f.write(f"Std Free Energy: {np.nanstd(F):.4f} kJ/mol\n")

                    finite_count = np.sum(np.isfinite(F))
                    total_count = F.size
                    f.write(f"Finite Bins: {finite_count} / {total_count} ({finite_count/total_count*100:.2f}%)\n")
                    f.write(f"Empty Bins: {total_count - finite_count} ({(total_count-finite_count)/total_count*100:.2f}%)\n\n")

                if hasattr(fes_obj, "xedges") and hasattr(fes_obj, "yedges"):
                    f.write(f"X Edges: {len(fes_obj.xedges)} bins\n")
                    f.write(f"  Range: [{fes_obj.xedges[0]:.4f}, {fes_obj.xedges[-1]:.4f}]\n")
                    f.write(f"Y Edges: {len(fes_obj.yedges)} bins\n")
                    f.write(f"  Range: [{fes_obj.yedges[0]:.4f}, {fes_obj.yedges[-1]:.4f}]\n\n")

                if hasattr(fes_obj, "metadata") and fes_obj.metadata:
                    f.write("FES Metadata:\n")
                    try:
                        serialized = _serialize_for_json(fes_obj.metadata)
                        f.write(json.dumps(serialized, indent=2))
                    except Exception as e:
                        f.write(f"<Error serializing metadata: {e}>\n")
                        f.write(str(fes_obj.metadata))
                    f.write("\n\n")

                if br.artifacts and "fes_quality" in br.artifacts:
                    f.write("FES Quality Metrics:\n")
                    try:
                        quality = _serialize_for_json(br.artifacts["fes_quality"])
                        f.write(json.dumps(quality, indent=2))
                    except Exception as e:
                        f.write(f"<Error serializing quality: {e}>\n")
                        f.write(str(br.artifacts["fes_quality"]))
                    f.write("\n")

                if br.feature_names:
                    f.write(f"\nCV Names: {', '.join(br.feature_names)}\n")
            else:
                f.write("No FES data available\n")

        logger.debug(f"Saved FES metrics to: {fes_file}")

    def _log_diagnostics(self, artifact: BuildArtifact, metrics_dir: Path) -> None:
        """Log diagnostic metrics."""
        diag_file = metrics_dir / "diagnostics.txt"

        with open(diag_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DIAGNOSTICS\n")
            f.write("=" * 80 + "\n\n")

            br = artifact.build_result
            if br and hasattr(br, "diagnostics") and br.diagnostics:
                diagnostics = br.diagnostics

                if "taus" in diagnostics:
                    f.write(f"Lag Times Used: {diagnostics['taus']}\n\n")

                if "diag_mass" in diagnostics and diagnostics["diag_mass"] is not None:
                    f.write(f"Diagonal Mass: {diagnostics['diag_mass']:.6f}\n\n")

                if "warnings" in diagnostics and diagnostics["warnings"]:
                    f.write("Warnings:\n")
                    for warning in diagnostics["warnings"]:
                        f.write(f"  - {warning}\n")
                    f.write("\n")

                if "canonical_correlations" in diagnostics:
                    f.write("Canonical Correlations:\n")
                    corrs = diagnostics["canonical_correlations"]
                    if isinstance(corrs, (list, np.ndarray)):
                        for i, corr in enumerate(corrs):
                            f.write(f"  Component {i}: {corr:.6f}\n")
                    f.write("\n")

                if "autocorrelation" in diagnostics:
                    f.write("Autocorrelation Data Available: Yes\n\n")

                f.write("Full Diagnostics (JSON):\n")
                try:
                    serialized = _serialize_for_json(diagnostics)
                    f.write(json.dumps(serialized, indent=2, default=str))
                except Exception as e:
                    f.write(f"<Error serializing diagnostics: {e}>\n")
                    f.write(str(diagnostics))
                f.write("\n")
            else:
                f.write("No diagnostics available\n")

            if artifact.guardrail_violations:
                f.write("\n" + "=" * 80 + "\n")
                f.write("GUARDRAIL VIOLATIONS\n")
                f.write("=" * 80 + "\n\n")
                for violation in artifact.guardrail_violations:
                    if isinstance(violation, dict):
                        try:
                            serialized = _serialize_for_json(violation)
                            f.write(json.dumps(serialized, indent=2))
                        except Exception:
                            f.write(str(violation))
                    else:
                        f.write(str(violation))
                    f.write("\n")

        logger.debug(f"Saved diagnostics to: {diag_file}")

    def _log_artifacts(self, artifact: BuildArtifact, metrics_dir: Path) -> None:
        """Log artifact details."""
        artifacts_file = metrics_dir / "artifacts.txt"

        with open(artifacts_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ARTIFACTS\n")
            f.write("=" * 80 + "\n\n")

            br = artifact.build_result
            if br and hasattr(br, "artifacts") and br.artifacts:
                for key, value in br.artifacts.items():
                    f.write(f"\n{key}:\n")
                    f.write("-" * 80 + "\n")

                    if isinstance(value, dict):
                        try:
                            serialized = _serialize_for_json(value)
                            f.write(json.dumps(serialized, indent=2, default=str))
                        except Exception as e:
                            f.write(f"<Error serializing: {e}>\n")
                            f.write(str(value))
                    elif isinstance(value, (list, tuple)):
                        try:
                            serialized = _serialize_for_json(value)
                            f.write(json.dumps(serialized, indent=2, default=str))
                        except Exception as e:
                            f.write(f"<Error serializing: {e}>\n")
                            f.write(str(value))
                    elif isinstance(value, np.ndarray):
                        f.write(f"NumPy Array: shape={value.shape}, dtype={value.dtype}\n")
                        if value.size < 100:
                            f.write(str(value))
                        else:
                            f.write(f"Array too large to display (size={value.size})")
                    else:
                        f.write(str(value))
                    f.write("\n\n")
            else:
                f.write("No artifacts available\n")

            if artifact.discretizer_fingerprint:
                f.write("\n" + "=" * 80 + "\n")
                f.write("DISCRETIZER FINGERPRINT\n")
                f.write("=" * 80 + "\n\n")
                try:
                    serialized = _serialize_for_json(artifact.discretizer_fingerprint)
                    f.write(json.dumps(serialized, indent=2, default=str))
                except Exception as e:
                    f.write(f"<Error serializing: {e}>\n")
                    f.write(str(artifact.discretizer_fingerprint))
                f.write("\n")

        logger.debug(f"Saved artifacts to: {artifacts_file}")

    def _log_build_summary(self, artifact: BuildArtifact, metrics_dir: Path) -> None:
        """Log overall build summary."""
        summary_file = metrics_dir / "build_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MSM/FES BUILD SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Build ID: {metrics_dir.name}\n")
            f.write(f"Bundle Path: {artifact.bundle_path}\n")
            f.write(f"Dataset Hash: {artifact.dataset_hash}\n")
            f.write(f"Created At: {artifact.created_at}\n")
            f.write(f"Analysis Healthy: {artifact.analysis_healthy}\n\n")

            br = artifact.build_result
            if br:
                f.write(f"Number of Frames: {br.n_frames}\n")
                f.write(f"Number of Shards: {br.n_shards}\n")

                if br.feature_names:
                    f.write(f"Feature Names: {', '.join(br.feature_names)}\n")

                if br.transition_matrix is not None:
                    f.write(f"MSM States: {br.transition_matrix.shape[0]}\n")

                if br.fes is not None:
                    f.write("FES: Available\n")
                else:
                    f.write("FES: Not Available\n")

                if br.diagnostics:
                    f.write("Diagnostics: Available\n")
                else:
                    f.write("Diagnostics: Not Available\n")

                if br.messages:
                    f.write("\nBuild Messages:\n")
                    for msg in br.messages:
                        f.write(f"  - {msg}\n")

                if br.flags:
                    f.write("\nBuild Flags:\n")
                    for key, value in br.flags.items():
                        f.write(f"  {key}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Metrics files created:\n")
            f.write("  - msm_metrics.txt\n")
            f.write("  - fes_metrics.txt\n")
            f.write("  - diagnostics.txt\n")
            f.write("  - artifacts.txt\n")
            f.write("  - build_summary.txt\n")
            f.write("=" * 80 + "\n")

        logger.debug(f"Saved build summary to: {summary_file}")

