from __future__ import annotations

from typing import Any

import streamlit as st


_DBSCAN_METRIC_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Estimator default (Euclidean)"),
    ("euclidean", "Euclidean (L2)"),
    ("manhattan", "Manhattan (L1)"),
    ("chebyshev", "Chebyshev (L∞)"),
    ("minkowski", "Minkowski (p-norm)"),
    ("cosine", "Cosine"),
)
_DBSCAN_ALGORITHM_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("ball_tree", "Ball tree"),
    ("kd_tree", "KD-tree"),
    ("brute", "Brute force"),
)


def _optional_int_input(
    container,
    *,
    label: str,
    key: str,
    default: int,
    min_value: int | None = None,
    step: int = 1,
    help_text: str | None = None,
) -> int | None:
    toggle_key = f"{key}_enabled"
    enabled = container.checkbox(
        f"Custom {label.lower()}",
        value=bool(st.session_state.get(toggle_key, False)),
        key=toggle_key,
    )
    if not enabled:
        return None
    current = int(st.session_state.get(key, default))
    return int(
        container.number_input(
            label,
            min_value=min_value,
            value=current,
            step=step,
            key=key,
            help=help_text,
        )
    )


def _optional_float_input(
    container,
    *,
    label: str,
    key: str,
    default: float,
    min_value: float | None = None,
    step: float = 0.1,
    fmt: str = "%.4f",
    help_text: str | None = None,
) -> float | None:
    toggle_key = f"{key}_enabled"
    enabled = container.checkbox(
        f"Custom {label.lower()}",
        value=bool(st.session_state.get(toggle_key, False)),
        key=toggle_key,
    )
    if not enabled:
        return None
    current = float(st.session_state.get(key, default))
    return float(
        container.number_input(
            label,
            min_value=min_value,
            value=current,
            step=step,
            format=fmt,
            key=key,
            help=help_text,
        )
    )


def render_dbscan_controls(*, prefix: str, min_samples_default: int = 5) -> dict[str, Any]:
    """Render DBSCAN parameter controls and return the kwargs dictionary."""

    container = st.container()
    container.markdown("**DBSCAN parameters**")
    kwargs: dict[str, Any] = {}

    auto_eps_key = f"{prefix}dbscan_auto_eps"
    auto_eps = container.checkbox(
        "Auto-select epsilon radius",
        value=bool(st.session_state.get(auto_eps_key, True)),
        key=auto_eps_key,
        help="Estimate eps from the reduced features using nearest-neighbour statistics.",
    )
    eps_value: float | None = None
    if not auto_eps:
        eps_key = f"{prefix}dbscan_eps"
        eps_value = float(
            container.number_input(
                "Neighborhood radius (eps)",
                min_value=1e-6,
                value=float(st.session_state.get(eps_key, 0.5)),
                step=0.05,
                format="%.4f",
                key=eps_key,
                help="Radius that defines the local neighbourhood for clustering.",
            )
        )

    min_samples_key = f"{prefix}dbscan_min_samples"
    min_samples = int(
        container.number_input(
            "Minimum samples",
            min_value=1,
            value=int(st.session_state.get(min_samples_key, min_samples_default)),
            step=1,
            key=min_samples_key,
            help="Points with at least this many neighbours are considered core samples.",
        )
    )

    metric_key = f"{prefix}dbscan_metric"
    metric_choice = container.selectbox(
        "Distance metric",
        options=[opt[0] for opt in _DBSCAN_METRIC_OPTIONS],
        format_func=lambda key: dict(_DBSCAN_METRIC_OPTIONS)[key],
        key=metric_key,
    )

    minkowski_p: float | None = None
    if metric_choice == "minkowski":
        minkowski_key = f"{prefix}dbscan_p"
        minkowski_p = float(
            container.number_input(
                "Minkowski power (p)",
                min_value=1.0,
                value=float(st.session_state.get(minkowski_key, 2.0)),
                step=0.5,
                key=minkowski_key,
            )
        )

    algorithm_key = f"{prefix}dbscan_algorithm"
    algorithm_choice = container.selectbox(
        "Neighbour search algorithm",
        options=[opt[0] for opt in _DBSCAN_ALGORITHM_OPTIONS],
        format_func=lambda key: dict(_DBSCAN_ALGORITHM_OPTIONS)[key],
        key=algorithm_key,
    )

    leaf_key = f"{prefix}dbscan_leaf_size"
    leaf_size = int(
        container.number_input(
            "Leaf size",
            min_value=5,
            value=int(st.session_state.get(leaf_key, 30)),
            step=5,
            key=leaf_key,
            help="Leaf size for Ball Tree or KD Tree searches.",
        )
    )

    jobs_choice_key = f"{prefix}dbscan_jobs_choice"
    jobs_choice = container.selectbox(
        "Parallel execution",
        options=["auto", "-1", "custom"],
        format_func=lambda choice: {
            "auto": "Estimator default (single core)",
            "-1": "All available cores (-1)",
            "custom": "Custom n_jobs",
        }[choice],
        key=jobs_choice_key,
    )
    n_jobs: int | None = None
    if jobs_choice == "-1":
        n_jobs = -1
    elif jobs_choice == "custom":
        jobs_value_key = f"{prefix}dbscan_n_jobs"
        n_jobs = int(
            container.number_input(
                "n_jobs",
                min_value=-1,
                value=int(st.session_state.get(jobs_value_key, 1)),
                step=1,
                key=jobs_value_key,
                help="Parallel workers for neighbourhood queries.",
            )
        )

    kwargs["min_samples"] = min_samples
    kwargs["leaf_size"] = leaf_size
    if eps_value is not None:
        kwargs["eps"] = eps_value
    if metric_choice != "auto":
        kwargs["metric"] = metric_choice
    if minkowski_p is not None:
        kwargs["p"] = minkowski_p
    if algorithm_choice != "auto":
        kwargs["algorithm"] = algorithm_choice
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs

    return kwargs


def render_kmeans_controls(*, prefix: str, allow_minibatch: bool = False) -> dict[str, Any]:
    """Render K-means/MiniBatchKMeans controls and return kwargs."""

    container = st.container()
    container.markdown("**K-means parameters**")
    kwargs: dict[str, Any] = {}

    max_iter = _optional_int_input(
        container,
        label="Max iterations",
        key=f"{prefix}kmeans_max_iter",
        default=300,
        min_value=1,
        help_text="Override the number of Lloyd iterations performed by the solver.",
    )
    if max_iter is not None:
        kwargs["max_iter"] = max_iter

    tolerance = _optional_float_input(
        container,
        label="Convergence tolerance",
        key=f"{prefix}kmeans_tol",
        default=1e-4,
        min_value=1e-8,
        step=1e-4,
        fmt="%.1e",
        help_text="Stop iterating when centroid updates fall below this threshold.",
    )
    if tolerance is not None:
        kwargs["tol"] = tolerance

    init_choice_key = f"{prefix}kmeans_init_choice"
    init_choice = container.selectbox(
        "Initialisation strategy",
        options=["default", "kmeans++", "random", "custom"],
        format_func=lambda key: {
            "default": "Estimator default (k-means++)",
            "kmeans++": "k-means++",
            "random": "Random",
            "custom": "Custom string",
        }[key],
        key=init_choice_key,
    )
    if init_choice == "custom":
        init_value_key = f"{prefix}kmeans_init_strategy"
        custom_value = container.text_input(
            "Custom init strategy",
            value=str(st.session_state.get(init_value_key, "")),
            key=init_value_key,
        ).strip()
        if custom_value:
            kwargs["init"] = custom_value
    elif init_choice != "default":
        kwargs["init"] = init_choice

    if allow_minibatch:
        batch_size = _optional_int_input(
            container,
            label="Mini-batch size",
            key=f"{prefix}kmeans_batch_size",
            default=1024,
            min_value=10,
            step=10,
            help_text="Number of samples processed per iteration when using MiniBatchKMeans.",
        )
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

    return kwargs


__all__ = ["render_dbscan_controls", "render_kmeans_controls"]
