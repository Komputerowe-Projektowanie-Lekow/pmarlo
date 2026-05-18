"""Active-bias retraining experiment on the Muller-Brown potential.

This script backs the Colab notebook
``14_muller_brown_active_bias_colab.ipynb``.  It is intentionally standalone:
it uses only NumPy, pandas, and matplotlib so it can run in Colab without a
Poetry install.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG_SEED = 20260518

MB_A = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
MB_a = np.array([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
MB_b = np.array([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
MB_c = np.array([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
MB_x0 = np.array([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
MB_y0 = np.array([0.0, 0.5, 1.5, 1.0], dtype=np.float64)

WALL_X_MIN, WALL_X_MAX = -1.5, 1.5
WALL_Y_MIN, WALL_Y_MAX = -0.5, 2.5
WALL_K = 1000.0

LANGEVIN_DT = 0.001
LANGEVIN_GAMMA = 5.0
LANGEVIN_KT = 15.0
LANGEVIN_MASS = 1.0

HILL_HEIGHT = 1.0
HILL_SIGMA = 0.1
HILL_STRIDE = 500

MB_INIT_BASIN = np.array([-0.55, 1.45], dtype=np.float64)
REFERENCE_GRID_BINS = 80

STATIONARY_POINTS = np.array(
    [
        [-0.558, 1.442],
        [0.623, 0.028],
        [-0.050, 0.467],
        [0.212, 0.293],
        [-0.822, 0.624],
    ],
    dtype=np.float64,
)
STATIONARY_ENERGIES = np.array([-146.7, -108.2, -80.8, -72.2, -40.7], dtype=np.float64)
BASIN_CENTERS = STATIONARY_POINTS[:3]
BASIN_LABELS = np.array(["A", "B", "C"])


@dataclass(frozen=True)
class ExperimentConfig:
    budget_frames: int
    replicates_per_condition: int
    lag: int
    vamp_components: int
    monitor_window: int
    monitor_stride: int
    train_window: int
    fixed_trigger_period: int
    threshold_delta: float
    adwin_delta: float
    adwin_min_window: int
    adwin_max_window: int
    max_pairs_per_epoch: int
    early_stopping_patience: int
    early_stopping_eps: float
    on_retrain_policy: str
    hill_height: float
    hill_sigma: float
    hill_stride: int
    trigger_policies: tuple[str, ...]
    data_policies: tuple[str, ...]
    training_policies: dict[str, dict[str, int | bool]]
    output_dir: Path


@dataclass(frozen=True)
class LinearCVModel2D:
    mean: np.ndarray
    projection: np.ndarray
    train_vamp2: float
    validation_vamp2: float
    epochs_run: int
    train_seconds: float

    def transform(self, xy: np.ndarray) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim == 1:
            return (arr - self.mean) @ self.projection
        return (arr - self.mean[None, :]) @ self.projection

    def jacobian(self, xy: np.ndarray | None = None) -> np.ndarray:
        return self.projection


@dataclass(frozen=True)
class Condition:
    trigger: str
    data_policy: str
    training_policy: str


def muller_brown_terms(x: float, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(x) - MB_x0
    dy = float(y) - MB_y0
    q = MB_a * dx * dx + MB_b * dx * dy + MB_c * dy * dy
    exp_q = np.exp(np.clip(q, -700.0, 80.0))
    return dx, dy, MB_A * exp_q


def muller_brown_potential(x: float, y: float) -> float:
    _, _, terms = muller_brown_terms(x, y)
    return float(np.sum(terms))


def muller_brown_force(x: float, y: float) -> np.ndarray:
    dx, dy, terms = muller_brown_terms(x, y)
    dqx = 2.0 * MB_a * dx + MB_b * dy
    dqy = MB_b * dx + 2.0 * MB_c * dy
    grad_x = float(np.sum(terms * dqx))
    grad_y = float(np.sum(terms * dqy))
    return np.array([-grad_x, -grad_y], dtype=np.float64)


def wall_force(x: float, y: float) -> np.ndarray:
    fx = 0.0
    fy = 0.0
    if x < WALL_X_MIN:
        fx += WALL_K * (WALL_X_MIN - x)
    elif x > WALL_X_MAX:
        fx -= WALL_K * (x - WALL_X_MAX)
    if y < WALL_Y_MIN:
        fy += WALL_K * (WALL_Y_MIN - y)
    elif y > WALL_Y_MAX:
        fy -= WALL_K * (y - WALL_Y_MAX)
    return np.array([fx, fy], dtype=np.float64)


def assert_muller_brown_stationary_energies() -> None:
    computed = np.array(
        [muller_brown_potential(x, y) for x, y in STATIONARY_POINTS],
        dtype=np.float64,
    )
    if not np.allclose(computed, STATIONARY_ENERGIES, atol=0.15):
        raise AssertionError(
            f"Unexpected Muller-Brown stationary energies: {computed.tolist()}"
        )


def assign_basin(xy: np.ndarray) -> str:
    distances = np.linalg.norm(np.asarray(xy, dtype=np.float64) - BASIN_CENTERS, axis=1)
    return str(BASIN_LABELS[int(np.argmin(distances))])


def mb_reference_probability() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xedges = np.linspace(WALL_X_MIN, WALL_X_MAX, REFERENCE_GRID_BINS + 1)
    yedges = np.linspace(WALL_Y_MIN, WALL_Y_MAX, REFERENCE_GRID_BINS + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xcenters, ycenters, indexing="ij")
    potential = np.vectorize(muller_brown_potential)(xx, yy)
    shifted = potential - float(np.min(potential))
    prob = np.exp(-shifted / LANGEVIN_KT)
    prob += 1.0e-300
    prob /= float(np.sum(prob))
    return prob.astype(np.float64), xedges, yedges


def _normalise_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray:
    if weights is None:
        return np.full(n_samples, 1.0 / float(n_samples), dtype=np.float64)
    raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    if raw.shape[0] != n_samples:
        raise ValueError("Weights must match the number of samples")
    if np.any(raw < 0.0) or not np.all(np.isfinite(raw)):
        raise ValueError("Weights must be finite and non-negative")
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value")
    return raw / total


def _inverse_sqrt(matrix: np.ndarray, reg: float = 1.0e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, reg)
    return (vectors / np.sqrt(values)) @ vectors.T


def vamp2_score_pairs(
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray | None,
    components: int,
) -> float:
    arr0 = np.asarray(x0, dtype=np.float64)
    arr1 = np.asarray(x1, dtype=np.float64)
    if arr0.ndim == 1:
        arr0 = arr0.reshape(-1, 1)
    if arr1.ndim == 1:
        arr1 = arr1.reshape(-1, 1)
    if arr0.shape != arr1.shape:
        raise ValueError("VAMP pairs must have matching shapes")
    if arr0.shape[0] < max(components + 2, 8):
        return float("nan")
    w = _normalise_weights(weights, arr0.shape[0])
    mean0 = np.average(arr0, axis=0, weights=w)
    mean1 = np.average(arr1, axis=0, weights=w)
    c0 = arr0 - mean0
    c1 = arr1 - mean1
    c00 = (c0 * w[:, None]).T @ c0
    c11 = (c1 * w[:, None]).T @ c1
    c01 = (c0 * w[:, None]).T @ c1
    koopman = _inverse_sqrt(c00) @ c01 @ _inverse_sqrt(c11)
    singular_values = np.linalg.svd(koopman, compute_uv=False)
    return float(np.sum(singular_values[:components] ** 2))


def vamp2_score_window(points: np.ndarray, cfg: ExperimentConfig) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] <= cfg.lag + 8:
        return float("nan")
    return vamp2_score_pairs(
        arr[: -cfg.lag],
        arr[cfg.lag :],
        weights=None,
        components=cfg.vamp_components,
    )


def _fit_projection(
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray | None,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    w = _normalise_weights(weights, x0.shape[0])
    mean0 = np.average(x0, axis=0, weights=w)
    mean1 = np.average(x1, axis=0, weights=w)
    c0 = x0 - mean0
    c1 = x1 - mean1
    c00 = (c0 * w[:, None]).T @ c0
    c11 = (c1 * w[:, None]).T @ c1
    c01 = (c0 * w[:, None]).T @ c1
    left = _inverse_sqrt(c00)
    right = _inverse_sqrt(c11)
    koopman = left @ c01 @ right
    u, _, _ = np.linalg.svd(koopman, full_matrices=False)
    return mean0, left @ u[:, :components]


def fit_linear_cv_model(
    points: np.ndarray,
    weights: np.ndarray | None,
    policy_name: str,
    cfg: ExperimentConfig,
) -> LinearCVModel2D:
    params = cfg.training_policies[policy_name]
    max_epochs = int(params["max_epochs"])
    early_stopping = bool(params["early_stopping"])
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy points with shape (n, 2), got {arr.shape}")
    if arr.shape[0] <= cfg.lag + 32:
        raise ValueError("Training window is too short for the configured lag")

    x0_all = arr[: -cfg.lag]
    x1_all = arr[cfg.lag :]
    pair_weights = None if weights is None else np.asarray(weights[: -cfg.lag])
    split = max(16, int(0.8 * x0_all.shape[0]))
    x0_train, x1_train = x0_all[:split], x1_all[:split]
    x0_val, x1_val = x0_all[split:], x1_all[split:]
    w_train = None if pair_weights is None else pair_weights[:split]
    w_val = None if pair_weights is None else pair_weights[split:]

    best_model: LinearCVModel2D | None = None
    last_model: LinearCVModel2D | None = None
    best_val = -np.inf
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        end = min(x0_train.shape[0], epoch * cfg.max_pairs_per_epoch)
        if end < 16:
            continue
        train_slice = slice(0, end)
        w_epoch = None if w_train is None else w_train[train_slice]
        mean, projection = _fit_projection(
            x0_train[train_slice],
            x1_train[train_slice],
            w_epoch,
            cfg.vamp_components,
        )
        train_cv0 = (x0_train[train_slice] - mean) @ projection
        train_cv1 = (x1_train[train_slice] - mean) @ projection
        val_cv0 = (x0_val - mean) @ projection
        val_cv1 = (x1_val - mean) @ projection
        train_score = vamp2_score_pairs(
            train_cv0,
            train_cv1,
            w_epoch,
            cfg.vamp_components,
        )
        val_score = vamp2_score_pairs(val_cv0, val_cv1, w_val, cfg.vamp_components)
        candidate = LinearCVModel2D(
            mean,
            projection,
            train_score,
            val_score,
            epoch,
            time.perf_counter() - start,
        )
        last_model = candidate
        if early_stopping:
            if val_score > best_val + cfg.early_stopping_eps:
                best_model = candidate
                best_val = val_score
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_model = candidate
        if early_stopping and stale_epochs >= cfg.early_stopping_patience:
            break

    if best_model is None:
        best_model = last_model
    if best_model is None:
        raise RuntimeError("Linear CV fitting did not produce a model")
    return best_model


class SimpleADWIN:
    def __init__(self, delta: float, min_window: int, max_window: int) -> None:
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self.window: deque[float] = deque()

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        while len(self.window) > self.max_window:
            self.window.popleft()
        if len(self.window) < 2 * self.min_window:
            return False
        values = np.asarray(self.window, dtype=np.float64)
        n_values = values.size
        for cut in range(self.min_window, n_values - self.min_window + 1):
            left = values[:cut]
            right = values[cut:]
            eps = math.sqrt(
                0.5 * math.log(4.0 / self.delta) * (1.0 / left.size + 1.0 / right.size)
            )
            if abs(float(left.mean() - right.mean())) > eps:
                for _ in range(cut):
                    self.window.popleft()
                return True
        return False


class ActiveBiasLedger:
    def __init__(self, sigma: float, height: float) -> None:
        self.sigma = float(sigma)
        self.height = float(height)
        self.centers_xy: list[np.ndarray] = []
        self.centers_cv: list[np.ndarray] = []

    def add_hill(self, xy: np.ndarray, cv_model: LinearCVModel2D) -> None:
        xy_arr = np.asarray(xy, dtype=np.float64).reshape(2)
        cv = np.asarray(cv_model.transform(xy_arr), dtype=np.float64).reshape(-1)
        self.centers_xy.append(xy_arr.copy())
        self.centers_cv.append(cv.copy())

    def reproject_to(self, cv_model: LinearCVModel2D) -> None:
        self.centers_cv = [
            np.asarray(cv_model.transform(xy), dtype=np.float64).reshape(-1)
            for xy in self.centers_xy
        ]

    def potential_in_cv(self, cv_value: np.ndarray) -> float:
        cv = np.asarray(cv_value, dtype=np.float64).reshape(-1)
        if not self.centers_cv:
            return 0.0
        centers = np.vstack(self.centers_cv)
        diff = cv[None, :] - centers
        r2 = np.sum(diff * diff, axis=1)
        gaussians = np.exp(-0.5 * r2 / (self.sigma * self.sigma))
        return float(self.height * np.sum(gaussians))

    def force_on_xy(self, xy: np.ndarray, cv_model: LinearCVModel2D) -> np.ndarray:
        if not self.centers_cv:
            return np.zeros(2, dtype=np.float64)
        xy_arr = np.asarray(xy, dtype=np.float64).reshape(2)
        cv = np.asarray(cv_model.transform(xy_arr), dtype=np.float64).reshape(-1)
        centers = np.vstack(self.centers_cv)
        diff = cv[None, :] - centers
        r2 = np.sum(diff * diff, axis=1)
        gaussians = np.exp(-0.5 * r2 / (self.sigma * self.sigma))
        d_v_d_cv = np.sum(
            (-self.height / (self.sigma * self.sigma)) * diff * gaussians[:, None],
            axis=0,
        )
        return -cv_model.jacobian(xy_arr) @ d_v_d_cv


class LangevinMB:
    def __init__(self, xy: np.ndarray, velocity: np.ndarray, rng: np.random.Generator):
        self.xy = np.asarray(xy, dtype=np.float64).reshape(2).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).reshape(2).copy()
        self.rng = rng
        self.c1 = math.exp(-LANGEVIN_GAMMA * LANGEVIN_DT)
        self.c2 = math.sqrt(1.0 - self.c1 * self.c1)

    def total_force(
        self,
        xy: np.ndarray,
        external_force_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        x_coord, y_coord = float(xy[0]), float(xy[1])
        force = (
            muller_brown_force(x_coord, y_coord)
            + wall_force(x_coord, y_coord)
            + np.asarray(external_force_fn(xy), dtype=np.float64)
        )
        if not np.all(np.isfinite(force)):
            raise FloatingPointError(f"Non-finite force at {xy}: {force}")
        return force

    def step(self, external_force_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        force = self.total_force(self.xy, external_force_fn)
        self.velocity += 0.5 * LANGEVIN_DT * force / LANGEVIN_MASS
        self.xy += 0.5 * LANGEVIN_DT * self.velocity
        noise = self.rng.normal(size=2)
        self.velocity = (
            self.c1 * self.velocity
            + self.c2 * math.sqrt(LANGEVIN_KT / LANGEVIN_MASS) * noise
        )
        self.xy += 0.5 * LANGEVIN_DT * self.velocity
        force_new = self.total_force(self.xy, external_force_fn)
        self.velocity += 0.5 * LANGEVIN_DT * force_new / LANGEVIN_MASS
        if not np.all(np.isfinite(self.xy)):
            raise FloatingPointError(f"Non-finite position: {self.xy}")
        return self.xy.copy()


def condition_seed(condition: Condition) -> int:
    text = f"{condition.trigger}|{condition.data_policy}|{condition.training_policy}"
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def stable_reweighting_factors(bias_values: np.ndarray) -> np.ndarray:
    raw = np.asarray(bias_values, dtype=np.float64) / LANGEVIN_KT
    shifted = raw - float(np.max(raw))
    weights = np.exp(shifted)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        raise ValueError("Invalid reweighting factors")
    return weights


def select_training_data(
    stream_xy: np.ndarray,
    bias_values: np.ndarray,
    end_frame: int,
    data_policy: str,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    if data_policy in {"Full", "Reweighted-Full"}:
        start = 0
    elif data_policy in {"Window-W", "Reweighted-Window"}:
        start = max(0, end_frame - cfg.train_window)
    else:
        raise ValueError(f"Unknown data policy: {data_policy}")
    points = np.asarray(stream_xy[start:end_frame], dtype=np.float64)
    weights = None
    if data_policy.startswith("Reweighted"):
        weights = stable_reweighting_factors(
            np.asarray(bias_values[start:end_frame], dtype=np.float64)
        )
    return points, weights


def should_trigger(
    condition: Condition,
    frame_index: int,
    last_train_frame: int,
    current_score: float,
    score_at_last_train: float,
    detector: SimpleADWIN,
    cfg: ExperimentConfig,
) -> bool:
    if condition.trigger == "Fixed-T":
        return frame_index - last_train_frame >= cfg.fixed_trigger_period
    if condition.trigger == "Threshold-delta":
        if not np.isfinite(score_at_last_train):
            return False
        return score_at_last_train - current_score > cfg.threshold_delta
    if condition.trigger == "ADWIN":
        return detector.update(current_score)
    raise ValueError(f"Unknown trigger: {condition.trigger}")


def run_unbiased_initialization(seed: int, steps: int = 2_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    velocity = rng.normal(scale=math.sqrt(LANGEVIN_KT / LANGEVIN_MASS), size=2)
    sim = LangevinMB(MB_INIT_BASIN.copy(), velocity, rng)
    zero_force = lambda xy: np.zeros(2, dtype=np.float64)
    trajectory = np.empty((steps, 2), dtype=np.float64)
    for frame in range(steps):
        trajectory[frame] = sim.step(zero_force)
    return trajectory


def make_bias_force_fn(
    ledger: ActiveBiasLedger,
    model: LinearCVModel2D,
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda xy: ledger.force_on_xy(np.asarray(xy, dtype=np.float64), model)


def histogram_probability_xy(
    points: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    weights: np.ndarray | None = None,
    pseudocount: float = 1.0e-12,
) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(xedges, yedges),
        weights=weights,
    )
    hist = hist.astype(np.float64, copy=False) + pseudocount
    hist /= float(np.sum(hist))
    return hist


def kl_divergence(p_ref: np.ndarray, p_est: np.ndarray) -> float:
    return float(np.sum(p_ref * (np.log(p_ref) - np.log(p_est))))


def coverage_fraction_xy(points: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(xedges, yedges))
    return float(np.count_nonzero(hist)) / float(hist.size)


def first_passage_times(
    transitions: list[tuple[int, str, str]],
) -> tuple[float, float]:
    first_ab = float("nan")
    first_ac = float("nan")
    for frame, old, new in transitions:
        if old == "A" and new == "B" and not np.isfinite(first_ab):
            first_ab = float(frame)
        if old == "A" and new == "C" and not np.isfinite(first_ac):
            first_ac = float(frame)
    return first_ab, first_ac


def independent_test_vamp2(model: LinearCVModel2D, seed: int, cfg: ExperimentConfig):
    trajectory = run_unbiased_initialization(seed, steps=4_000)
    cv = model.transform(trajectory)
    return vamp2_score_window(cv, cfg)


def metrics_mb(
    condition: Condition,
    replica: int,
    trajectory_xy: np.ndarray,
    collection_bias_values: np.ndarray,
    ledger: ActiveBiasLedger,
    retrain_events: list[int],
    basin_transitions: list[tuple[int, str, str]],
    model: LinearCVModel2D,
    total_train_seconds: float,
    monitor_scores: list[tuple[int, float]],
    cfg: ExperimentConfig,
    reference_probability: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> dict[str, float | int | str]:
    collection_weights = stable_reweighting_factors(collection_bias_values)
    final_bias = np.array(
        [ledger.potential_in_cv(model.transform(xy)) for xy in trajectory_xy],
        dtype=np.float64,
    )
    final_weights = stable_reweighting_factors(final_bias)
    p_collection = histogram_probability_xy(
        trajectory_xy,
        xedges,
        yedges,
        weights=collection_weights,
    )
    p_final = histogram_probability_xy(
        trajectory_xy,
        xedges,
        yedges,
        weights=final_weights,
    )
    p_raw = histogram_probability_xy(trajectory_xy, xedges, yedges)
    first_ab, first_ac = first_passage_times(basin_transitions)
    transition_pairs = {(old, new) for _, old, new in basin_transitions if old != new}
    score_values = np.array([score for _, score in monitor_scores], dtype=np.float64)
    test_seed = RNG_SEED + 90_000 + replica + condition_seed(condition)

    return {
        **asdict(condition),
        "replica": replica,
        "budget_frames": cfg.budget_frames,
        "kl_ref_to_collection_reweighted": kl_divergence(
            reference_probability,
            p_collection,
        ),
        "kl_ref_to_final_reweighted": kl_divergence(reference_probability, p_final),
        "kl_ref_to_raw": kl_divergence(reference_probability, p_raw),
        "coverage_xy": coverage_fraction_xy(trajectory_xy, xedges, yedges),
        "retrain_count": len(retrain_events),
        "transition_count": len(basin_transitions),
        "unique_transition_count": len(transition_pairs),
        "first_A_to_B": first_ab,
        "first_A_to_C": first_ac,
        "total_train_seconds": total_train_seconds,
        "final_train_vamp2": model.train_vamp2,
        "final_validation_vamp2": model.validation_vamp2,
        "final_epochs_run": model.epochs_run,
        "test_vamp2": independent_test_vamp2(model, test_seed, cfg),
        "monitor_vamp2_mean": (
            float(np.nanmean(score_values)) if score_values.size else float("nan")
        ),
        "monitor_vamp2_std": (
            float(np.nanstd(score_values)) if score_values.size else float("nan")
        ),
        "final_hill_count": len(ledger.centers_cv),
        "on_retrain_policy": cfg.on_retrain_policy,
    }


def run_mb_condition_replica(
    condition: Condition,
    replica: int,
    cfg: ExperimentConfig,
    reference_probability: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> dict[str, float | int | str]:
    seed = RNG_SEED + 10_000 * replica + condition_seed(condition)
    rng = np.random.default_rng(seed)
    velocity = rng.normal(scale=math.sqrt(LANGEVIN_KT / LANGEVIN_MASS), size=2)
    sim = LangevinMB(MB_INIT_BASIN.copy(), velocity, rng)

    init_xy = run_unbiased_initialization(seed + 1_000)
    model = fit_linear_cv_model(init_xy, None, condition.training_policy, cfg)
    total_train_seconds = model.train_seconds
    ledger = ActiveBiasLedger(cfg.hill_sigma, cfg.hill_height)
    bias_force_fn = make_bias_force_fn(ledger, model)

    trajectory_xy = np.empty((cfg.budget_frames, 2), dtype=np.float64)
    cv_value_buffer: list[np.ndarray] = []
    bias_value_buffer = np.empty(cfg.budget_frames, dtype=np.float64)
    monitor_scores: list[tuple[int, float]] = []
    retrain_events: list[int] = []
    basin_transitions: list[tuple[int, str, str]] = []
    last_train_frame = 0
    score_at_last_train = float("nan")
    detector = SimpleADWIN(
        cfg.adwin_delta,
        cfg.adwin_min_window,
        cfg.adwin_max_window,
    )
    previous_basin = assign_basin(MB_INIT_BASIN)

    for frame in range(cfg.budget_frames):
        xy = sim.step(bias_force_fn)
        cv = np.asarray(model.transform(xy), dtype=np.float64).reshape(-1)
        trajectory_xy[frame] = xy
        cv_value_buffer.append(cv)
        bias_value_buffer[frame] = ledger.potential_in_cv(cv)

        if frame % cfg.hill_stride == 0:
            ledger.add_hill(xy, model)

        basin = assign_basin(xy)
        if basin != previous_basin:
            basin_transitions.append((frame, previous_basin, basin))
            previous_basin = basin

        if frame % cfg.monitor_stride == 0 and frame > cfg.monitor_window:
            window_cv = np.asarray(cv_value_buffer[-cfg.monitor_window :])
            score = vamp2_score_window(window_cv, cfg)
            monitor_scores.append((frame, score))
            if np.isfinite(score) and should_trigger(
                condition,
                frame,
                last_train_frame,
                score,
                score_at_last_train,
                detector,
                cfg,
            ):
                train_points, train_weights = select_training_data(
                    trajectory_xy[:frame],
                    bias_value_buffer[:frame],
                    frame,
                    condition.data_policy,
                    cfg,
                )
                model = fit_linear_cv_model(
                    train_points,
                    train_weights,
                    condition.training_policy,
                    cfg,
                )
                total_train_seconds += model.train_seconds
                if cfg.on_retrain_policy == "reset_ledger":
                    ledger = ActiveBiasLedger(cfg.hill_sigma, cfg.hill_height)
                elif cfg.on_retrain_policy == "reproject_centers":
                    ledger.reproject_to(model)
                else:
                    raise ValueError(
                        f"Unknown on_retrain_policy: {cfg.on_retrain_policy}"
                    )
                bias_force_fn = make_bias_force_fn(ledger, model)
                retrain_events.append(frame)
                last_train_frame = frame
                score_at_last_train = score

    return metrics_mb(
        condition,
        replica,
        trajectory_xy,
        bias_value_buffer,
        ledger,
        retrain_events,
        basin_transitions,
        model,
        total_train_seconds,
        monitor_scores,
        cfg,
        reference_probability,
        xedges,
        yedges,
    )


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "kl_ref_to_collection_reweighted",
        "kl_ref_to_final_reweighted",
        "kl_ref_to_raw",
        "coverage_xy",
        "retrain_count",
        "transition_count",
        "unique_transition_count",
        "total_train_seconds",
        "test_vamp2",
        "final_validation_vamp2",
        "final_epochs_run",
    ]
    summary = results_df.groupby(
        ["trigger", "data_policy", "training_policy"],
        as_index=False,
    )[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(col).rstrip("_") for col in summary.columns.to_flat_index()
    ]
    return summary.sort_values(
        [
            "kl_ref_to_collection_reweighted_mean",
            "coverage_xy_mean",
            "unique_transition_count_mean",
        ],
        ascending=[True, False, False],
    )


def plot_results(summary: pd.DataFrame, output_dir: Path) -> Path:
    top = summary.head(12).copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    labels = (
        top["trigger"] + " | " + top["data_policy"] + " | " + top["training_policy"]
    )

    axes[0].barh(range(top.shape[0]), top["kl_ref_to_collection_reweighted_mean"])
    axes[0].set_yticks(range(top.shape[0]), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("KL(ref || reweighted), mean")
    axes[0].set_title("Best MB KL strategies")

    scatter = axes[1].scatter(
        summary["retrain_count_mean"],
        summary["coverage_xy_mean"],
        c=summary["kl_ref_to_collection_reweighted_mean"],
        cmap="viridis_r",
    )
    axes[1].set_xlabel("retraining events")
    axes[1].set_ylabel("coverage in (x, y)")
    axes[1].set_title("Coverage vs retraining overhead")
    fig.colorbar(scatter, ax=axes[1], label="KL")

    axes[2].scatter(
        summary["unique_transition_count_mean"],
        summary["test_vamp2_mean"],
        s=50 + 20 * summary["retrain_count_mean"],
        alpha=0.8,
    )
    axes[2].set_xlabel("unique basin transitions")
    axes[2].set_ylabel("independent unbiased test VAMP-2")
    axes[2].set_title("CV quality vs transition discovery")

    fig.tight_layout()
    plot_path = output_dir / "muller_brown_active_bias_plots.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def write_protocol(
    cfg: ExperimentConfig,
    results_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> Path:
    protocol = {
        "source": "analytic Muller-Brown potential with active metadynamics-like bias",
        "scope": "active feedback test: bias force enters the Langevin BAOAB integrator",
        "budget_frames": cfg.budget_frames,
        "replicates_per_condition": cfg.replicates_per_condition,
        "lag": cfg.lag,
        "monitor_window": cfg.monitor_window,
        "monitor_stride": cfg.monitor_stride,
        "train_window": cfg.train_window,
        "fixed_trigger_period": cfg.fixed_trigger_period,
        "threshold_delta": cfg.threshold_delta,
        "adwin_delta": cfg.adwin_delta,
        "muller_brown": {
            "A": MB_A.tolist(),
            "a": MB_a.tolist(),
            "b": MB_b.tolist(),
            "c": MB_c.tolist(),
            "x0": MB_x0.tolist(),
            "y0": MB_y0.tolist(),
        },
        "walls": {
            "x_min": WALL_X_MIN,
            "x_max": WALL_X_MAX,
            "y_min": WALL_Y_MIN,
            "y_max": WALL_Y_MAX,
            "k": WALL_K,
        },
        "langevin": {
            "dt": LANGEVIN_DT,
            "gamma": LANGEVIN_GAMMA,
            "kT": LANGEVIN_KT,
            "mass": LANGEVIN_MASS,
        },
        "bias": {
            "height": cfg.hill_height,
            "sigma": cfg.hill_sigma,
            "stride": cfg.hill_stride,
            "mode": "classical",
            "on_retrain_policy": cfg.on_retrain_policy,
        },
        "training_policies": cfg.training_policies,
        "outputs": {
            "raw_results": str(results_path),
            "summary": str(summary_path),
            "plot": str(plot_path),
        },
    }
    protocol_path = cfg.output_dir / "muller_brown_active_bias_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol_path


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        hill_height = args.hill_height if args.hill_height is not None else 2.0
        training_policies: dict[str, dict[str, int | bool]] = {
            "Fixed-10ep": {"max_epochs": 10, "early_stopping": False},
            "EarlyStopping-20ep": {"max_epochs": 20, "early_stopping": True},
        }
        return ExperimentConfig(
            budget_frames=args.budget_frames or 6_000,
            replicates_per_condition=args.replicates or 2,
            lag=10,
            vamp_components=1,
            monitor_window=800,
            monitor_stride=500,
            train_window=2_000,
            fixed_trigger_period=1_500,
            threshold_delta=0.08,
            adwin_delta=0.01,
            adwin_min_window=12,
            adwin_max_window=80,
            max_pairs_per_epoch=2_048,
            early_stopping_patience=8,
            early_stopping_eps=1.0e-4,
            on_retrain_policy=args.on_retrain_policy,
            hill_height=hill_height,
            hill_sigma=args.hill_sigma,
            hill_stride=args.hill_stride,
            trigger_policies=("Fixed-T", "ADWIN"),
            data_policies=("Window-W", "Reweighted-Window"),
            training_policies=training_policies,
            output_dir=output_dir,
        )

    hill_height = args.hill_height if args.hill_height is not None else HILL_HEIGHT
    training_policies = {
        "Fixed-50ep": {"max_epochs": 50, "early_stopping": False},
        "Fixed-200ep": {"max_epochs": 200, "early_stopping": False},
        "EarlyStopping": {"max_epochs": 200, "early_stopping": True},
    }
    return ExperimentConfig(
        budget_frames=args.budget_frames or 25_000,
        replicates_per_condition=args.replicates or 10,
        lag=10,
        vamp_components=1,
        monitor_window=2_000,
        monitor_stride=500,
        train_window=8_000,
        fixed_trigger_period=5_000,
        threshold_delta=0.08,
        adwin_delta=0.01,
        adwin_min_window=12,
        adwin_max_window=80,
        max_pairs_per_epoch=2_048,
        early_stopping_patience=8,
        early_stopping_eps=1.0e-4,
        on_retrain_policy=args.on_retrain_policy,
        hill_height=hill_height,
        hill_sigma=args.hill_sigma,
        hill_stride=args.hill_stride,
        trigger_policies=("Fixed-T", "Threshold-delta", "ADWIN"),
        data_policies=("Full", "Window-W", "Reweighted-Full", "Reweighted-Window"),
        training_policies=training_policies,
        output_dir=output_dir,
    )


def run_experiment(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    assert_muller_brown_stationary_energies()
    if cfg.hill_height <= 0.0:
        raise ValueError("hill_height must be positive")
    if cfg.hill_sigma <= 0.0:
        raise ValueError("hill_sigma must be positive")
    if cfg.hill_stride <= 0:
        raise ValueError("hill_stride must be positive")
    reference_probability, xedges, yedges = mb_reference_probability()
    conditions = [
        Condition(trigger, data_policy, training_policy)
        for trigger, data_policy, training_policy in itertools.product(
            cfg.trigger_policies,
            cfg.data_policies,
            cfg.training_policies.keys(),
        )
    ]

    results: list[dict[str, float | int | str]] = []
    total_runs = len(conditions) * cfg.replicates_per_condition
    run_index = 0
    started = time.perf_counter()
    for replica in range(cfg.replicates_per_condition):
        for condition in conditions:
            run_index += 1
            print(f"[{run_index:03d}/{total_runs:03d}] {condition} replica={replica}")
            results.append(
                run_mb_condition_replica(
                    condition,
                    replica,
                    cfg,
                    reference_probability,
                    xedges,
                    yedges,
                )
            )

    results_df = pd.DataFrame(results)
    results_path = cfg.output_dir / "muller_brown_active_bias_results.csv"
    results_df.to_csv(results_path, index=False)
    summary = aggregate_results(results_df)
    summary_path = cfg.output_dir / "muller_brown_active_bias_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_path = plot_results(summary, cfg.output_dir)
    protocol_path = write_protocol(cfg, results_path, summary_path, plot_path)

    print(f"Saved raw results to {results_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {plot_path}")
    print(f"Saved protocol to {protocol_path}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.1f}")
    print(summary.head(12).to_string(index=False))
    return results_df, summary, protocol_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a small Colab-friendly grid.",
    )
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--budget-frames", type=int, default=None)
    parser.add_argument(
        "--on-retrain-policy",
        choices=("reset_ledger", "reproject_centers"),
        default="reset_ledger",
    )
    parser.add_argument(
        "--hill-height",
        type=float,
        default=None,
        help=(
            "Metadynamics hill height. Defaults to 2.0 in quick mode and 1.0 "
            "in the full Muller-Brown protocol."
        ),
    )
    parser.add_argument("--hill-sigma", type=float, default=HILL_SIGMA)
    parser.add_argument("--hill-stride", type=int, default=HILL_STRIDE)
    parser.add_argument(
        "--output-dir",
        default=(
            Path("example_programs")
            / "programs_outputs"
            / "muller_brown_active_bias_colab"
        ),
    )
    return parser.parse_args()


def main() -> None:
    cfg = build_config(parse_args())
    print(f"Output directory: {cfg.output_dir}")
    print(
        "Conditions:",
        len(cfg.trigger_policies) * len(cfg.data_policies) * len(cfg.training_policies),
        "x",
        cfg.replicates_per_condition,
        "replicas",
    )
    expected_hills = math.ceil(cfg.budget_frames / cfg.hill_stride)
    print(
        "Bias budget:",
        f"{expected_hills} hills x {cfg.hill_height} =",
        f"{expected_hills * cfg.hill_height:.1f} energy units",
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
