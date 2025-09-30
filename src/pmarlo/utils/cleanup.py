from __future__ import annotations

"""
Space management utilities for PMARLO workspaces and runs.

This module provides safe, opt-in helpers to prune large intermediate files
once shards are emitted and bundles are built. It does not remove any files by
default; all functions support ``dry_run=True`` to list what would be removed.

Typical usage
-------------
- After running the sharded app:

    from pathlib import Path
    from pmarlo.utils.cleanup import prune_workspace

    # List candidates only (no deletion)
    report = prune_workspace(Path("example_programs/app_usecase/app_output"), mode="conservative", dry_run=True)
    print(report)

    # Delete large DCDs (per-replica + demux) and checkpoints, keep shards/bundles
    report = prune_workspace(Path("example_programs/app_usecase/app_output"), mode="conservative", dry_run=False)

Policies
--------
- conservative: remove per-replica DCDs, demuxed DCDs (keep .meta.json), checkpoint dirs,
  progress logs, and feature_cache directories under runs.
- aggressive: conservative + remove model artifacts older than 7 days and redundant bundles
  (keeps newest N=3 by mtime) under the workspace. Use with care.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

PruneMode = Literal["conservative", "aggressive"]


@dataclass
class PruneReport:
    root: Path
    removed: List[Path] = field(default_factory=list)
    kept: List[Path] = field(default_factory=list)
    errors: List[Tuple[Path, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "root": [str(self.root)],
            "removed": [str(p) for p in self.removed],
            "kept": [str(p) for p in self.kept],
            "errors": [f"{p}: {msg}" for p, msg in self.errors],
        }


def _iter_runs(root: Path) -> Iterable[Path]:
    """Yield run directories under a workspace root.

    Recognizes app_usecase layout: ``root / sims / run-*``.
    Falls back to any directory under ``root`` containing ``replica_exchange``.
    """
    sims = root / "sims"
    if sims.exists():
        for p in sorted(sims.glob("run-*")):
            if (p / "replica_exchange").exists():
                yield p
    else:
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "replica_exchange").exists():
                yield p


def _collect_candidates(run_dir: Path, keep_demux_meta: bool = True) -> List[Path]:
    """Collect deletion candidates inside a single run directory."""
    cand: List[Path] = []
    rx = run_dir / "replica_exchange"
    if rx.exists():
        # Per-replica DCDs
        cand.extend(sorted(rx.glob("replica_*.dcd")))
        # Demuxed DCDs (keep sidecar metadata by default)
        for d in sorted(rx.glob("demux_T*K.dcd")):
            cand.append(d)
            if not keep_demux_meta:
                meta = d.with_suffix(".meta.json")
                if meta.exists():
                    cand.append(meta)
        # Checkpoints
        for ck in (rx / "checkpoints", run_dir / "checkpoints"):
            if ck.exists():
                cand.append(ck)
    # Progress logs
    logs = (
        run_dir.parent.parent / "logs"
        if run_dir.parent.name == "sims"
        else run_dir / "logs"
    )
    if logs.exists():
        cand.extend(sorted(logs.glob("*.log")))
    # Feature caches under this run (if any)
    fc = run_dir / "feature_cache"
    if fc.exists():
        cand.append(fc)
    return cand


def _safe_remove(path: Path, report: PruneReport) -> None:
    try:
        if path.is_dir():
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        report.removed.append(path)
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        report.errors.append((path, str(exc)))


def prune_workspace(
    root: Path, *, mode: PruneMode = "conservative", dry_run: bool = True
) -> PruneReport:
    """Prune large intermediates from a PMARLO workspace.

    Parameters
    ----------
    root : Path
        Workspace root, e.g., ``example_programs/app_usecase/app_output``.
    mode : {"conservative", "aggressive"}
        - conservative: remove per-replica and demuxed DCDs, checkpoints, logs, feature_cache.
        - aggressive: also prune old model artifacts and keep only newest 3 bundles.
    dry_run : bool
        When True, only report candidates without deleting.

    Returns
    -------
    PruneReport
        Summary of removed/kept paths and any errors encountered.
    """
    root = Path(root)
    rep = PruneReport(root=root)

    for run in _iter_runs(root):
        if not _has_downstream_artifacts(root, run):
            rep.kept.append(run)
            continue
        _prune_run_candidates(run, dry_run, rep)

    if mode == "aggressive":
        _prune_old_bundles(root, dry_run, rep)
        _prune_stale_models(root, dry_run, rep)

    return rep


def _has_downstream_artifacts(root: Path, run: Path) -> bool:
    shards_dir = root / "shards" / run.name
    bundles_dir = root / "bundles"
    have_shards = shards_dir.exists() and any(shards_dir.glob("*.json"))
    have_bundles = bundles_dir.exists() and any(bundles_dir.glob("*.json"))
    return have_shards or have_bundles


def _prune_run_candidates(run: Path, dry_run: bool, report: PruneReport) -> None:
    for candidate in _collect_candidates(run, keep_demux_meta=True):
        if dry_run:
            report.kept.append(candidate)
        else:
            _safe_remove(candidate, report)


def _prune_old_bundles(root: Path, dry_run: bool, report: PruneReport) -> None:
    bundles_dir = root / "bundles"
    if not bundles_dir.exists():
        return
    bundles = sorted(
        bundles_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in bundles[3:]:
        if dry_run:
            report.kept.append(old)
        else:
            _safe_remove(old, report)


def _prune_stale_models(root: Path, dry_run: bool, report: PruneReport) -> None:
    import time

    models_root = root / "models"
    if not models_root.exists():
        return
    cutoff = time.time() - 7 * 24 * 3600
    for artifact in models_root.rglob("*"):
        try:
            if not artifact.is_file() or artifact.stat().st_mtime >= cutoff:
                continue
        except Exception:
            continue
        if dry_run:
            report.kept.append(artifact)
        else:
            _safe_remove(artifact, report)
