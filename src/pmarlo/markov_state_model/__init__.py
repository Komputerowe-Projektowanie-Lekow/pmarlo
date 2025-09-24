"""Public exports for the Markov state model toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "MarkovStateModel",
    "run_complete_msm_analysis",
    "run_ck",
    "CKRunResult",
    "FESResult",
    "PMFResult",
    "generate_1d_pmf",
    "generate_2d_fes",
    "periodic_kde_2d",
    "pca_reduce",
    "tica_reduce",
    "vamp_reduce",
    "reduce_features",
    "get_available_methods",
    "BaseResult",
    "REMDResult",
    "DemuxResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    "Reweighter",
    "MSMBuilder",
    "BuilderMSMResult",
]

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "MarkovStateModel": (
        "pmarlo.markov_state_model.enhanced_msm",
        "EnhancedMSM",
    ),
    "run_complete_msm_analysis": (
        "pmarlo.markov_state_model.enhanced_msm",
        "run_complete_msm_analysis",
    ),
    "run_ck": ("pmarlo.markov_state_model.ck_runner", "run_ck"),
    "CKRunResult": ("pmarlo.markov_state_model.ck_runner", "CKRunResult"),
    "FESResult": ("pmarlo.markov_state_model.free_energy", "FESResult"),
    "PMFResult": ("pmarlo.markov_state_model.free_energy", "PMFResult"),
    "generate_1d_pmf": ("pmarlo.markov_state_model.free_energy", "generate_1d_pmf"),
    "generate_2d_fes": ("pmarlo.markov_state_model.free_energy", "generate_2d_fes"),
    "periodic_kde_2d": ("pmarlo.markov_state_model.free_energy", "periodic_kde_2d"),
    "pca_reduce": ("pmarlo.markov_state_model.reduction", "pca_reduce"),
    "tica_reduce": ("pmarlo.markov_state_model.reduction", "tica_reduce"),
    "vamp_reduce": ("pmarlo.markov_state_model.reduction", "vamp_reduce"),
    "reduce_features": ("pmarlo.markov_state_model.reduction", "reduce_features"),
    "get_available_methods": (
        "pmarlo.markov_state_model.reduction",
        "get_available_methods",
    ),
    "BaseResult": ("pmarlo.markov_state_model.results", "BaseResult"),
    "CKResult": ("pmarlo.markov_state_model.results", "CKResult"),
    "ClusteringResult": ("pmarlo.markov_state_model.results", "ClusteringResult"),
    "DemuxResult": ("pmarlo.markov_state_model.results", "DemuxResult"),
    "ITSResult": ("pmarlo.markov_state_model.results", "ITSResult"),
    "MSMResult": ("pmarlo.markov_state_model.results", "MSMResult"),
    "REMDResult": ("pmarlo.markov_state_model.results", "REMDResult"),
    "Reweighter": ("pmarlo.markov_state_model.reweighter", "Reweighter"),
    "MSMBuilder": ("pmarlo.markov_state_model.msm_builder", "MSMBuilder"),
    "BuilderMSMResult": ("pmarlo.markov_state_model.msm_builder", "MSMResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        module = import_module(module_name)
    except Exception as exc:
        value: Any
        if name == "run_ck":  # pragma: no cover - executed without matplotlib

            def _missing_run_ck(
                *_args: object, original_exc=exc, **_kwargs: object
            ) -> None:
                raise ImportError(
                    "run_ck requires matplotlib. Install with `pip install 'pmarlo[analysis]'`."
                ) from original_exc

            value = _missing_run_ck
        elif name == "CKRunResult":  # pragma: no cover - executed without matplotlib

            class _CKRunResult:  # type: ignore[override]
                pass

            value = _CKRunResult
        else:  # pragma: no cover - defensive guard for other optional exports
            raise
    else:
        value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
