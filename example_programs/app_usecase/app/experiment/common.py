from __future__ import annotations

"""Shared helpers for the PMARLO experiment workflows (E0/E1/E2).

This module centralises filesystem layout, configuration loading, and shard
resolution so the runnable scripts can focus on orchestration logic. The user
provides actual shard data; we merely validate that expected manifests and
config entries are present.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml

from pmarlo.data.aggregate import load_shards_as_dataset
from pmarlo.shards.format import read_shard
from pmarlo.shards.schema import Shard

__all__ = [
    "ExperimentBundle",
    "ExperimentConfigs",
    "ExperimentLayout",
    "ExperimentPaths",
    "ExperimentRegistry",
    "ExperimentConfigError",
    "load_bundle",
    "list_experiments",
]

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentLayout:
    """Declarative mapping between experiment name and directory layout."""

    name: str
    shard_dirname: str
    output_dirname: str
    optional: bool = False


EXPERIMENT_LAYOUTS: tuple[ExperimentLayout, ...] = (
    ExperimentLayout(
        name="E0_same_temp",
        shard_dirname="same_temp_shards",
        output_dirname="same_temp_output",
        optional=False,
    ),
    ExperimentLayout(
        name="E1_mixed_ladders",
        shard_dirname="mixed_ladders_shards",
        output_dirname="mixed_ladders_output",
        optional=False,
    ),
    ExperimentLayout(
        name="E2_disjoint_ladders",
        shard_dirname="disjoint_ladders_shards",
        output_dirname="disjoint_ladders_output",
        optional=True,
    ),
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved filesystem paths for experiment assets."""

    app_root: Path
    inputs_root: Path
    configs_dir: Path
    outputs_root: Path

    @classmethod
    def from_app_root(cls, app_root: Path | None = None) -> "ExperimentPaths":
        base = Path(app_root or Path(__file__).resolve().parent)
        inputs_root = (base.parent / "app_intputs" / "experiments").resolve()
        outputs_root = (base / "experiment_outputs").resolve()
        configs_dir = (inputs_root / "configs").resolve()
        return cls(
            app_root=base.resolve(),
            inputs_root=inputs_root,
            configs_dir=configs_dir,
            outputs_root=outputs_root,
        )

    def shards_dir(self, layout: ExperimentLayout) -> Path:
        return (self.inputs_root / layout.shard_dirname).resolve()

    def output_dir(self, layout: ExperimentLayout) -> Path:
        return (self.outputs_root / layout.output_dirname).resolve()


# --------------------------------------------------------------------------- #
# Config loading / utilities
# --------------------------------------------------------------------------- #


class ExperimentConfigError(RuntimeError):
    """Raised when configuration files are missing or inconsistent."""


def _read_yaml(path: Path) -> MutableMapping[str, Any]:
    if not path.exists():
        raise ExperimentConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise ExperimentConfigError(f"Unexpected YAML structure in {path}: {type(data)}")
    return data


def _deep_merge(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Recursively merge two mapping objects."""

    result: Dict[str, Any] = {}
    if base:
        for key, value in base.items():
            if isinstance(value, Mapping):
                result[key] = _deep_merge(value, None)
            else:
                result[key] = value
    if override:
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def _select_experiment_section(
    data: Mapping[str, Any],
    experiment_name: str,
    *,
    allow_common: bool = True,
) -> Dict[str, Any]:
    experiments = data.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ExperimentConfigError("Expected top-level 'experiments' mapping in config file.")
    selected = experiments.get(experiment_name)
    if not isinstance(selected, Mapping):
        raise ExperimentConfigError(f"Experiment '{experiment_name}' missing in config file.")
    common = data.get("common") if allow_common else None
    if common is not None and not isinstance(common, Mapping):
        raise ExperimentConfigError("Top-level 'common' section must be a mapping if present.")
    return _deep_merge(common, selected)


def _resolve_manifest(layout: ExperimentLayout, paths: ExperimentPaths) -> Path:
    manifest_path = paths.shards_dir(layout) / "manifest.yaml"
    if not manifest_path.exists():
        raise ExperimentConfigError(
            f"Manifest not found for experiment {layout.name}: {manifest_path}"
        )
    return manifest_path


def _resolve_shard_jsons(
    manifest: Mapping[str, Any], layout: ExperimentLayout, paths: ExperimentPaths
) -> tuple[Path, ...]:
    shards = manifest.get("shards")
    if not isinstance(shards, Iterable):
        raise ExperimentConfigError(f"Manifest for {layout.name} lacks 'shards' iterable.")

    shard_dir = paths.shards_dir(layout)
    resolved: list[Path] = []

    for entry in shards:
        if not isinstance(entry, Mapping):
            raise ExperimentConfigError(f"Invalid shard entry in manifest: {entry!r}")
        shard_id = entry.get("shard_id")
        if not shard_id:
            raise ExperimentConfigError("Shard manifest entry missing 'shard_id'.")
        json_name = entry.get("json") or f"{shard_id}.json"
        json_path = (shard_dir / str(json_name)).resolve()
        if not json_path.exists():
            raise ExperimentConfigError(f"Shard metadata JSON missing: {json_path}")
        resolved.append(json_path)
    if not resolved:
        raise ExperimentConfigError(f"Manifest for {layout.name} lists no shards.")
    return tuple(resolved)


@dataclass(frozen=True)
class ExperimentConfigs:
    """Aggregated configuration dictionaries for a single experiment."""

    transform: Mapping[str, Any]
    discretize: Mapping[str, Any]
    reweighter: Mapping[str, Any]
    msm: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentBundle:
    """Resolved experiment configuration and input assets."""

    layout: ExperimentLayout
    manifest_path: Path
    manifest: Mapping[str, Any]
    shard_jsons: tuple[Path, ...]
    configs: ExperimentConfigs
    output_dir: Path

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_shards(self) -> tuple[Shard, ...]:
        return tuple(read_shard(path) for path in self.shard_jsons)

    def load_dataset(self) -> Mapping[str, Any]:
        return load_shards_as_dataset(self.shard_jsons)

    @property
    def reference_summary_path(self) -> Path | None:
        refs = self.manifest.get("reference_summary")
        manifest_dir = self.manifest_path.parent
        if isinstance(refs, str):
            return (manifest_dir / refs).resolve()
        ref_dir = manifest_dir.parent / "refs" / "long_run_Tref"
        candidate = ref_dir / "summary.npz"
        return candidate if candidate.exists() else None


class ExperimentRegistry:
    """Registry that resolves experiment bundles based on the filesystem layout."""

    def __init__(self, paths: ExperimentPaths | None = None):
        self.paths = paths or ExperimentPaths.from_app_root()

    def get_layout(self, experiment_name: str) -> ExperimentLayout:
        for layout in EXPERIMENT_LAYOUTS:
            if layout.name == experiment_name:
                return layout
        raise ExperimentConfigError(f"Unknown experiment '{experiment_name}'.")

    def load_bundle(self, experiment_name: str) -> ExperimentBundle:
        layout = self.get_layout(experiment_name)
        manifest_path = _resolve_manifest(layout, self.paths)
        manifest = _read_yaml(manifest_path)
        shard_jsons = _resolve_shard_jsons(manifest, layout, self.paths)

        transform_cfg = _select_experiment_section(
            _read_yaml(self.paths.configs_dir / "transform_plan.yaml"),
            experiment_name,
            allow_common=False,
        )
        discretize_cfg = _select_experiment_section(
            _read_yaml(self.paths.configs_dir / "discretize.yaml"),
            experiment_name,
            allow_common=True,
        )
        reweighter_cfg = _select_experiment_section(
            _read_yaml(self.paths.configs_dir / "reweighter.yaml"),
            experiment_name,
            allow_common=False,
        )
        msm_cfg = _select_experiment_section(
            _read_yaml(self.paths.configs_dir / "msm.yaml"),
            experiment_name,
            allow_common=True,
        )

        configs = ExperimentConfigs(
            transform=transform_cfg,
            discretize=discretize_cfg,
            reweighter=reweighter_cfg,
            msm=msm_cfg,
        )
        output_dir = self.paths.output_dir(layout)
        return ExperimentBundle(
            layout=layout,
            manifest_path=manifest_path,
            manifest=manifest,
            shard_jsons=shard_jsons,
            configs=configs,
            output_dir=output_dir,
        )


def load_bundle(experiment_name: str, *, app_root: Path | None = None) -> ExperimentBundle:
    """Convenience wrapper returning the resolved ExperimentBundle."""

    registry = ExperimentRegistry(
        ExperimentPaths.from_app_root(app_root) if app_root else None
    )
    return registry.load_bundle(experiment_name)


def list_experiments(include_optional: bool = True) -> list[str]:
    """List registered experiment names."""

    if include_optional:
        return [layout.name for layout in EXPERIMENT_LAYOUTS]
    return [layout.name for layout in EXPERIMENT_LAYOUTS if not layout.optional]
