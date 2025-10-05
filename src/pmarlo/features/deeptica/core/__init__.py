"""Internal building blocks for DeepTICA training."""

from importlib import import_module
from typing import Any, Dict, Tuple

from .dataset import DatasetBundle, create_dataset, create_loaders, split_sequences
from .history import (
    LossHistory,
    collect_history_metrics,
    project_model,
    summarize_history,
    vamp2_proxy,
)
from .inputs import FeaturePrep, prepare_features
from .pairs import PairInfo, build_pair_info
from .utils import safe_float, set_all_seeds

__all__ = [
    "DatasetBundle",
    "create_dataset",
    "create_loaders",
    "split_sequences",
    "LossHistory",
    "collect_history_metrics",
    "project_model",
    "summarize_history",
    "vamp2_proxy",
    "FeaturePrep",
    "prepare_features",
    "PairInfo",
    "build_pair_info",
    "safe_float",
    "set_all_seeds",
]

_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "apply_output_whitening": (
        "pmarlo.features.deeptica.core.model",
        "apply_output_whitening",
    ),
    "build_network": ("pmarlo.features.deeptica.core.model", "build_network"),
    "TrainingArtifacts": (
        "pmarlo.features.deeptica.core.trainer_api",
        "TrainingArtifacts",
    ),
    "train_deeptica_pipeline": (
        "pmarlo.features.deeptica.core.trainer_api",
        "train_deeptica_pipeline",
    ),
}

__all__.extend(_OPTIONAL_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _OPTIONAL_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))
