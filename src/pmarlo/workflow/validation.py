"""Workflow validation utilities for PMARLO analysis pipelines.

This module provides comprehensive validation functions for checking data
consistency, shard usage, and analysis quality across the PMARLO workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from pmarlo.io.catalog import (
    ShardCatalog,
    build_catalog_from_paths,
)


def validate_build_result(
    build_result: Dict[str, Any],
    available_shard_paths: List[Path],
    dataset_hash: str = "",
) -> Dict[str, Any]:
    """
    Comprehensive validation of build results against available shards.

    This function performs thorough validation of the build process, checking
    for consistency between available shards and those used in the build,
    identifying data quality issues, and providing actionable diagnostics.

    Parameters
    ----------
    build_result : Dict[str, Any]
        Result dictionary from the build process containing artifacts
    available_shard_paths : List[Path]
        List of paths to all available shard trajectory files
    dataset_hash : str, optional
        Dataset hash for integrity verification

    Returns
    -------
    Dict[str, Any]
        Validation results with keys:
        - "is_valid": bool indicating if validation passed
        - "messages": List[str] of informational messages
        - "warnings": List[str] of warning messages
        - "errors": List[str] of error messages
        - "shard_table": List[Dict] with detailed shard information
        - "summary": Dict with usage statistics

    Examples
    --------
    >>> result = {"artifacts": {"shards_used": ["run1:demux:T300:0", "run1:demux:T350:1"]}}
    >>> paths = [Path("run1/demux_T300K.dcd"), Path("run1/demux_T350K.dcd")]
    >>> validation = validate_build_result(result, paths)
    >>> print(validation["is_valid"])
    True
    """
    validation_results: Dict[str, Any] = {
        "is_valid": True,
        "messages": [],
        "warnings": [],
        "errors": [],
        "shard_table": [],
        "summary": {},
    }

    # Extract used canonical IDs from build result
    used_ids = _extract_used_canonical_ids(build_result)
    validation_results["summary"]["used_shard_count"] = len(used_ids)

    # Build catalog from available paths
    catalog = build_catalog_from_paths(available_shard_paths, dataset_hash)
    validation_results["summary"]["available_shard_count"] = len(catalog.shards)

    # Validate usage consistency
    usage_validation = catalog.validate_against_used(used_ids)

    # Process missing shards
    if usage_validation["missing"]:
        validation_results["warnings"].append(
            f"Missing shards in build: {len(usage_validation['missing'])} shards not used"
        )
        for missing_id in usage_validation["missing"]:
            validation_results["warnings"].append(f"  - {missing_id}")

    # Process extra shards
    if usage_validation["extra"]:
        validation_results["warnings"].append(
            f"Extra shards in build: {len(usage_validation['extra'])} shards used but not found"
        )
        for extra_id in usage_validation["extra"]:
            validation_results["warnings"].append(f"  - {extra_id}")

    # Add catalog warnings
    validation_results["warnings"].extend(usage_validation["warnings"])

    # Generate shard information table
    validation_results["shard_table"] = catalog.get_shard_info_table()

    # Generate summary messages
    total_available = len(catalog.shards)
    total_used = len(used_ids)

    if (
        total_used == total_available
        and not usage_validation["missing"]
        and not usage_validation["extra"]
    ):
        validation_results["messages"].append(
            f"Build used all {total_used} available shards"
        )
    elif total_used < total_available:
        usage_ratio = total_used / total_available if total_available > 0 else 0
        validation_results["messages"].append(
            f"Build used {total_used}/{total_available} available shards ({usage_ratio:.1%})"
        )
    else:
        validation_results["messages"].append(
            f"Build used {total_used} shards ({total_available} available)"
        )

    # Mark as invalid if there are critical issues
    if usage_validation["extra"]:
        validation_results["is_valid"] = False
        validation_results["errors"].append(
            "Build references shards not present in available data"
        )

    # Additional quality checks
    quality_issues = _check_data_quality(build_result, catalog)
    validation_results["warnings"].extend(quality_issues)

    return validation_results


def validate_fes_quality(
    fes_data: Dict[str, Any], build_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate Free Energy Surface (FES) computation quality.

    This function analyzes FES data for common quality issues like empty bins,
    NaN values, and insufficient sampling.

    Parameters
    ----------
    fes_data : Dict[str, Any]
        FES computation results
    build_result : Dict[str, Any], optional
        Build result for additional context

    Returns
    -------
    Dict[str, Any]
        Validation results with quality metrics and warnings
    """
    validation_results: Dict[str, Any] = {
        "is_valid": True,
        "messages": [],
        "warnings": [],
        "errors": [],
        "metrics": {},
    }

    try:
        fes_array = _extract_fes_array(fes_data)
    except ValueError as e:
        validation_results["is_valid"] = False
        validation_results["errors"].append(str(e))
        return validation_results

    _evaluate_nan_metrics(fes_array, validation_results)
    _evaluate_empty_bins(fes_array, validation_results)
    _assess_fes_range(fes_array, validation_results)

    # Overall assessment
    validation_results["messages"].append("FES quality validation completed")

    return validation_results


def _extract_fes_array(fes_data: Dict[str, Any]) -> np.ndarray:
    """Return the numeric FES array from common artifact formats."""

    for key in ("fes", "values", "F"):
        if key in fes_data:
            return np.asarray(fes_data[key])
    raise ValueError("No FES values found in data")


def _evaluate_nan_metrics(
    fes_array: np.ndarray, validation_results: Dict[str, Any]
) -> None:
    nan_count = int(np.isnan(fes_array).sum())
    if nan_count <= 0 or fes_array.size == 0:
        return
    nan_ratio = nan_count / fes_array.size
    validation_results["warnings"].append(
        f"FES contains {nan_count} NaN values ({nan_ratio:.1%} of total)"
    )


def _evaluate_empty_bins(
    fes_array: np.ndarray, validation_results: Dict[str, Any]
) -> None:
    if not hasattr(fes_array, "shape") or len(fes_array.shape) != 2:
        return

    max_reasonable_energy = 100.0  # kT units
    empty_bins = int(np.sum(fes_array > max_reasonable_energy))
    if empty_bins <= 0 or fes_array.size == 0:
        return

    empty_ratio = empty_bins / fes_array.size
    validation_results["metrics"]["empty_bins_ratio"] = empty_ratio

    if empty_ratio >= 0.5:
        validation_results["warnings"].append(
            f"High fraction of empty FES bins ({empty_ratio:.1%}) - "
            "consider increasing sampling or adjusting bin ranges"
        )
    elif empty_ratio >= 0.1:
        validation_results["warnings"].append(
            f"empty FES bins detected ({empty_ratio:.1%}) - check sampling quality"
        )
    else:
        validation_results["messages"].append(
            f"Low empty FES bin ratio detected ({empty_ratio:.1%})"
        )


def _assess_fes_range(
    fes_array: np.ndarray, validation_results: Dict[str, Any]
) -> None:
    finite_values = fes_array[np.isfinite(fes_array)]
    if finite_values.size == 0:
        return

    fes_range = float(np.ptp(finite_values))  # peak-to-peak
    validation_results["metrics"]["fes_range"] = fes_range
    if fes_range < 1.0:
        validation_results["warnings"].append(
            f"Narrow FES range ({fes_range:.1f} kT) - check if data covers sufficient phase space"
        )


def _extract_used_canonical_ids(build_result: Dict[str, Any]) -> Set[str]:
    """
    Extract canonical shard IDs from build result artifacts.

    Parameters
    ----------
    build_result : Dict[str, Any]
        Build result dictionary

    Returns
    -------
    Set[str]
        Set of canonical IDs used in the build
    """
    used_ids = set()

    # Extract from artifacts
    artifacts = build_result.get("artifacts", {})
    shards_used = artifacts.get("shards_used", [])

    for shard_id in shards_used:
        if isinstance(shard_id, str):
            # Check if it's already a canonical ID
            if ":" in shard_id and len(shard_id.split(":")) == 4:
                used_ids.add(shard_id)
            else:
                raise ValueError(
                    f"Non-canonical shard identifier encountered: {shard_id}"
                )

    return used_ids


def _check_data_quality(
    build_result: Dict[str, Any], catalog: ShardCatalog
) -> List[str]:
    """
    Perform additional data quality checks.

    Parameters
    ----------
    build_result : Dict[str, Any]
        Build result dictionary
    catalog : ShardCatalog
        Shard catalog for analysis

    Returns
    -------
    List[str]
        List of quality warning messages
    """
    warnings = []

    # Check for single-run datasets
    if len(catalog.run_ids) == 1:
        run_id = list(catalog.run_ids)[0]
        warnings.append(f"Single run dataset: {run_id}")

    # Check temperature distribution
    demux_shards = [s for s in catalog.shards.values() if s.source_kind == "demux"]
    if demux_shards:
        temps = [s.temperature_K for s in demux_shards if s.temperature_K is not None]
        if temps:
            temp_range = max(temps) - min(temps)
            if temp_range < 100:  # Less than 100K range
                warnings.append(
                    f"Narrow temperature range: {min(temps)}K - {max(temps)}K. "
                    "Consider broader temperature sampling for better convergence."
                )

    # Check for bias information
    artifacts = build_result.get("artifacts", {})
    bias_info = artifacts.get("bias_artifacts", {})
    if not bias_info:
        warnings.append(
            "No bias information found - results may not reflect reweighting"
        )

    return warnings


def format_validation_report(validation_results: Dict[str, Any]) -> str:
    """
    Format validation results into a human-readable report.

    Parameters
    ----------
    validation_results : Dict[str, Any]
        Validation results from validate_build_result()

    Returns
    -------
    str
        Formatted validation report
    """
    lines = []

    # Status
    status = "VALID" if validation_results["is_valid"] else "INVALID"
    lines.append(f"Build Validation: {status}")
    lines.append("")

    # Summary
    summary = validation_results.get("summary", {})
    lines.append("Summary:")
    lines.append(f"  Available shards: {summary.get('available_shard_count', 'N/A')}")
    lines.append(f"  Used shards: {summary.get('used_shard_count', 'N/A')}")
    lines.append("")

    # Messages
    if validation_results["messages"]:
        lines.append("Messages:")
        for msg in validation_results["messages"]:
            lines.append(f"  {msg}")
        lines.append("")

    # Warnings
    if validation_results["warnings"]:
        lines.append("Warnings:")
        for warning in validation_results["warnings"]:
            lines.append(f"  ! {warning}")
        lines.append("")

    # Errors
    if validation_results["errors"]:
        lines.append("Errors:")
        for error in validation_results["errors"]:
            lines.append(f"  X {error}")
        lines.append("")

    # Shard table (first 10 entries)
    shard_table = validation_results.get("shard_table", [])
    if shard_table:
        lines.append("Shard Information (first 10):")
        lines.append(
            "  Canonical ID                    | Run ID          | Kind  | Temp/Index | Path"
        )
        lines.append("  " + "-" * 95)

        for i, shard in enumerate(shard_table[:10]):
            canonical = (
                shard["canonical_id"][:32] + "..."
                if len(shard["canonical_id"]) > 32
                else shard["canonical_id"]
            )
            run_id = (
                shard["run_id"][:15] + "..."
                if len(shard["run_id"]) > 15
                else shard["run_id"]
            )
            path = (
                shard["source_path"][-40:]
                if len(shard["source_path"]) > 40
                else shard["source_path"]
            )

            row_template = (
                "  {canonical:<32} | {run_id:<15} | {kind:<5} | {temp:<10} | {path}"
            )
            lines.append(
                row_template.format(
                    canonical=canonical,
                    run_id=run_id,
                    kind=shard["source_kind"],
                    temp=shard["temp_or_replica"],
                    path=path,
                )
            )

        if len(shard_table) > 10:
            lines.append(f"  ... and {len(shard_table) - 10} more shards")
        lines.append("")

    return "\n".join(lines)
