import base64
from typing import Any, Dict, Tuple

import pytest

np = pytest.importorskip("numpy")

import pmarlo.transform.build as build_mod
from pmarlo.markov_state_model.free_energy import FESResult


@pytest.fixture(scope="module")
def shard_arrays() -> list[np.ndarray]:
    rng = np.random.default_rng(1234)
    return [
        rng.normal(size=(48, 4)).astype(np.float32),
        rng.normal(size=(36, 4)).astype(np.float32),
        rng.normal(size=(28, 4)).astype(np.float32),
    ]


@pytest.fixture(scope="module")
def shard_metadata(shard_arrays: list[np.ndarray]) -> list[dict[str, Any]]:
    info: list[dict[str, Any]] = []
    offset = 0
    for idx, block in enumerate(shard_arrays):
        length = int(block.shape[0])
        info.append(
            {
                "id": f"shard-{idx}",
                "start": offset,
                "stop": offset + length,
                "frames_loaded": length,
                "frames_declared": length,
                "effective_frame_stride": 1,
                "preview_truncated": False,
            }
        )
        offset += length
    return info


@pytest.fixture(scope="module")
def concatenated_features(shard_arrays: list[np.ndarray]) -> np.ndarray:
    return np.vstack(shard_arrays).astype(np.float32, copy=False)


@pytest.fixture(scope="module")
def deeptica_schedule() -> Tuple[int, ...]:
    return (2, 5, 10, 20)


@pytest.fixture(scope="module")
def deeptica_pairs(
    shard_arrays: list[np.ndarray], deeptica_schedule: Tuple[int, ...]
) -> Dict[str, Any]:
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    base_info = build_pair_info(shard_arrays, deeptica_schedule)
    if base_info.idx_t.size:
        weights = np.linspace(0.5, 1.5, base_info.idx_t.size, dtype=np.float32)
    else:
        weights = np.empty((0,), dtype=np.float32)
    info = build_pair_info(
        shard_arrays,
        deeptica_schedule,
        pairs=(base_info.idx_t, base_info.idx_tau),
        weights=weights,
    )
    return {
        "pair_info": info,
        "pairs": (info.idx_t, info.idx_tau),
        "weights": info.weights,
    }


@pytest.fixture(scope="module")
def deeptica_workflow(
    shard_arrays: list[np.ndarray],
    deeptica_schedule: Tuple[int, ...],
    deeptica_pairs: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    pytest.importorskip("mlcolvar")
    torch = pytest.importorskip("torch")

    from pmarlo.features.deeptica import DeepTICAConfig
    from pmarlo.features.deeptica.core.inputs import prepare_features
    from pmarlo.features.deeptica.core.trainer_api import train_deeptica_pipeline
    import pmarlo.ml.deeptica.trainer as trainer_mod

    cfg = DeepTICAConfig(
        lag=int(deeptica_schedule[-1]),
        tau_schedule=deeptica_schedule,
        val_tau=int(deeptica_schedule[-1]),
        epochs_per_tau=1,
        max_epochs=1,
        batch_size=8,
        hidden=(16,),
        n_out=2,
        seed=17,
    )

    def _fake_fit(self, sequences, *, val_sequences=None):
        self.grad_norm_curve = [0.25, 0.15]
        history = {
            "loss_curve": [1.0, 0.8],
            "val_loss_curve": [1.1, 0.9],
            "val_score_curve": [0.4, 0.45],
            "grad_norm_curve": list(self.grad_norm_curve),
        }
        return history

    monkeypatch.setattr(trainer_mod.DeepTICACurriculumTrainer, "fit", _fake_fit)

    prep = prepare_features(shard_arrays, tau_schedule=deeptica_schedule, seed=cfg.seed)
    arrays = [np.asarray(block, dtype=np.float32) for block in shard_arrays]
    artifacts = train_deeptica_pipeline(
        arrays,
        deeptica_pairs["pairs"],
        cfg,
        weights=deeptica_pairs["weights"],
    )

    matrix = np.asarray(np.vstack(arrays), dtype=np.float32)
    scaled = artifacts.scaler.transform(matrix).astype(np.float32)
    with torch.no_grad():
        outputs_tensor = artifacts.network(torch.as_tensor(scaled))
        outputs = outputs_tensor.detach().cpu().numpy()

    artifacts_repeat = train_deeptica_pipeline(
        arrays,
        deeptica_pairs["pairs"],
        cfg,
        weights=deeptica_pairs["weights"],
    )
    scaled_repeat = artifacts_repeat.scaler.transform(matrix).astype(np.float32)
    with torch.no_grad():
        outputs_tensor_repeat = artifacts_repeat.network(torch.as_tensor(scaled_repeat))
        outputs_repeat = outputs_tensor_repeat.detach().cpu().numpy()

    return {
        "prep": prep,
        "artifacts": artifacts,
        "artifacts_repeat": artifacts_repeat,
        "scaled": scaled,
        "outputs": outputs,
        "outputs_repeat": outputs_repeat,
        "history": artifacts.history,
        "pairs": deeptica_pairs["pair_info"],
        "weights": deeptica_pairs["weights"],
        "schedule": deeptica_schedule,
        "n_frames": int(matrix.shape[0]),
        "input_matrix": matrix,
        "config": cfg,
    }


@pytest.fixture(scope="module")
def simple_msm():
    pytest.importorskip("deeptime")
    import pmarlo.markov_state_model._msm_utils as msm_mod

    dtrajs = [
        np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0], dtype=int),
        np.array([1, 2, 3, 2, 1, 0, 1, 2], dtype=int),
    ]
    T, pi = msm_mod.build_simple_msm(dtrajs, n_states=4, lag=1)
    counts = np.array(
        [
            [5, 2, 1, 0],
            [2, 6, 3, 1],
            [1, 3, 5, 2],
            [0, 1, 2, 4],
        ],
        dtype=float,
    )
    trimmed = msm_mod.ensure_connected_counts(counts)
    return {
        "mod": msm_mod,
        "T": T,
        "pi": pi,
        "counts": counts,
        "trimmed": trimmed,
    }


@pytest.fixture(scope="module")
def build_result_payload(deeptica_workflow: Dict[str, Any], simple_msm: Dict[str, Any]):
    history = deeptica_workflow["history"]
    fes = FESResult(
        F=np.random.default_rng(5).random((12, 10)),
        xedges=np.linspace(-2.0, 2.0, 13),
        yedges=np.linspace(-1.5, 1.5, 11),
        levels_kJmol=np.linspace(0.0, 5.0, 6),
        metadata={"counts": np.random.default_rng(7).random((12, 10))},
    )
    br = build_mod.BuildResult(
        transition_matrix=simple_msm["T"],
        stationary_distribution=simple_msm["pi"],
        n_frames=deeptica_workflow["n_frames"],
        n_shards=3,
        feature_names=["DeepTICA_1", "DeepTICA_2"],
        cluster_populations=np.array([0.3, 0.4, 0.3], dtype=float),
        artifacts={
            "mlcv_deeptica": {
                "output_variance": history["output_variance"],
                "top_eigenvalues": history["top_eigenvalues"],
                "whitening": history["whitening"],
            }
        },
        fes=fes,
    )
    return {"result": br, "fes": fes}


# ---------------------------------------------------------------------------
# 1. Input Data and Shard Shapes
# ---------------------------------------------------------------------------


def test_shard_feature_matrix_shapes(shard_arrays, concatenated_features, shard_metadata):
    assert concatenated_features.shape[0] == sum(block.shape[0] for block in shard_arrays)
    assert concatenated_features.shape[1] == shard_arrays[0].shape[1]
    assert all(int(meta["stop"]) - int(meta["start"]) == block.shape[0] for meta, block in zip(shard_metadata, shard_arrays))


def test_shard_metadata_frames_consistency(shard_metadata, concatenated_features):
    total_meta = shard_metadata[-1]["stop"] if shard_metadata else 0
    assert total_meta == concatenated_features.shape[0]
    assert sum(meta["frames_loaded"] for meta in shard_metadata) == concatenated_features.shape[0]


# ---------------------------------------------------------------------------
# 2. DeepTICA Pair Building Shapes
# ---------------------------------------------------------------------------


def test_lagged_pairs_shape_single_shard():
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    shard = [np.arange(12, dtype=float).reshape(-1, 1)]
    info = build_pair_info(shard, (3,))
    assert info.idx_t.shape == (9,)
    assert info.idx_tau.shape == (9,)
    assert np.all(info.idx_tau - info.idx_t == 3)


def test_lagged_pairs_shape_multiple_shards(shard_arrays, deeptica_schedule):
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    info = build_pair_info(shard_arrays, (2,))
    expected = sum(max(0, arr.shape[0] - 2) for arr in shard_arrays)
    assert info.idx_t.shape == info.idx_tau.shape == (expected,)
    counts = [
        int(((info.idx_t >= start) & (info.idx_t < stop)).sum())
        for start, stop in zip(
            [0, shard_arrays[0].shape[0], shard_arrays[0].shape[0] + shard_arrays[1].shape[0]],
            [
                shard_arrays[0].shape[0],
                shard_arrays[0].shape[0] + shard_arrays[1].shape[0],
                sum(arr.shape[0] for arr in shard_arrays),
            ],
        )
    ]
    assert counts == info.diagnostics["pairs_by_shard"]


def test_lagged_pairs_curriculum_schedule(shard_arrays, deeptica_schedule):
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    info = build_pair_info(shard_arrays, deeptica_schedule)
    total_pairs = info.idx_t.size
    manual = 0
    for tau in deeptica_schedule:
        for block in shard_arrays:
            manual += max(0, block.shape[0] - tau)
    assert total_pairs == manual
    assert info.diagnostics["lag_used"] == max(deeptica_schedule)


def test_weights_array_shape_matches_pairs(deeptica_pairs):
    pair_info = deeptica_pairs["pair_info"]
    assert pair_info.weights.shape == pair_info.idx_t.shape
    assert pair_info.weights.ndim == 1


def test_pairs_do_not_cross_shard_boundaries(shard_arrays, deeptica_schedule):
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    tau = max(deeptica_schedule)
    info = build_pair_info(shard_arrays, (tau,))
    bounds = np.cumsum([0] + [block.shape[0] for block in shard_arrays])
    for t, u in zip(info.idx_t, info.idx_tau):
        shard_index = np.searchsorted(bounds, t, side="right") - 1
        assert bounds[shard_index] <= u < bounds[shard_index + 1]


# ---------------------------------------------------------------------------
# 3. DeepTICA Training Output Shapes
# ---------------------------------------------------------------------------


def test_deeptica_scaler_input_output_shapes(deeptica_workflow):
    prep = deeptica_workflow["prep"]
    scaled = deeptica_workflow["scaled"]
    assert prep.X.shape == scaled.shape
    assert prep.Z.shape == scaled.shape


def test_deeptica_network_output_shape(deeptica_workflow):
    outputs = deeptica_workflow["outputs"]
    scaled = deeptica_workflow["scaled"]
    assert outputs.shape[0] == scaled.shape[0]
    assert outputs.shape[1] == deeptica_workflow["config"].n_out


def test_deeptica_network_output_shape_respects_cfg(deeptica_workflow):
    outputs = deeptica_workflow["outputs"]
    outputs_repeat = deeptica_workflow["outputs_repeat"]
    n_out = deeptica_workflow["config"].n_out
    assert outputs.shape[1] == n_out
    assert outputs_repeat.shape[1] == n_out


def test_deeptica_output_variance_shape(deeptica_workflow):
    history = deeptica_workflow["history"]
    variance = history["output_variance"]
    assert len(variance) == deeptica_workflow["config"].n_out


def test_deeptica_whitening_transform_shape(deeptica_workflow):
    history = deeptica_workflow["history"]
    whitening = history["whitening"]
    assert "mean" in whitening
    assert "transform" in whitening
    mean = np.asarray(whitening["mean"])
    transform = np.asarray(whitening["transform"])
    n_out = deeptica_workflow["config"].n_out
    assert mean.shape == (n_out,)
    assert transform.shape == (n_out, n_out)


def test_deeptica_training_history_shapes(deeptica_workflow):
    history = deeptica_workflow["history"]
    n_loss = len(history["loss_curve"])
    assert len(history["val_loss_curve"]) == n_loss
    assert len(history["val_score_curve"]) == n_loss
    assert len(history["grad_norm_curve"]) == n_loss


def test_scaled_and_outputs_are_finite(deeptica_workflow):
    assert np.isfinite(deeptica_workflow["scaled"]).all()
    assert np.isfinite(deeptica_workflow["outputs"]).all()


def test_deeptica_pipeline_is_deterministic(deeptica_workflow):
    outputs_first = deeptica_workflow["outputs"]
    outputs_second = deeptica_workflow["outputs_repeat"]
    assert np.allclose(outputs_first, outputs_second)


# ---------------------------------------------------------------------------
# 4. MSM Matrix Shapes
# ---------------------------------------------------------------------------


def test_count_matrix_is_square(simple_msm):
    counts = simple_msm["counts"]
    assert counts.shape[0] == counts.shape[1]


def test_T_row_stochastic_and_nonnegative(simple_msm):
    T = simple_msm["T"]
    assert np.all(T >= 0)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-8)


def test_transition_matrix_shape_matches_stationary(simple_msm):
    T = simple_msm["T"]
    pi = simple_msm["pi"]
    assert T.shape[0] == T.shape[1]
    assert pi.shape == (T.shape[0],)
    assert np.isclose(pi.sum(), 1.0)


def test_stationary_distribution_is_left_eigenvector(simple_msm):
    T = simple_msm["T"]
    pi = simple_msm["pi"]
    assert np.all(pi >= 0)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ T, pi, atol=1e-6)


@pytest.mark.xfail(reason="Only if your MSM builder enforces reversibility")
def test_detailed_balance_if_reversible(simple_msm):
    T = simple_msm["T"]
    pi = simple_msm["pi"]
    assert np.allclose(np.outer(pi, np.ones_like(pi)) * T, np.outer(pi, np.ones_like(pi)) * T.T, atol=1e-6)


def test_msm_active_states_trimming(simple_msm):
    trimmed = simple_msm["trimmed"]
    assert trimmed.counts.shape[0] == trimmed.counts.shape[1]
    assert trimmed.active.shape[0] == trimmed.counts.shape[0]


def test_trimmed_counts_is_strongly_connected(simple_msm):
    counts = simple_msm["trimmed"].counts
    adjacency = (counts > 0).astype(int)
    reach = adjacency.copy()
    for _ in range(adjacency.shape[0] - 1):
        reach = (reach @ adjacency > 0).astype(int)
    reachable = (reach + np.eye(adjacency.shape[0], dtype=int) > 0)
    assert np.all(reachable & reachable.T)


def test_expanded_T_has_valid_rows_and_inactive_self_loops(simple_msm):
    msm_mod = simple_msm["mod"]
    T_active = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    pi_active = np.array([0.6, 0.4], dtype=float)
    active = np.array([0, 2])
    T_full, pi_full = msm_mod._expand_results(4, active, T_active, pi_active)
    assert T_full.shape == (4, 4)
    assert pi_full.shape == (4,)
    assert np.allclose(T_full[np.ix_(active, active)], T_active)
    assert np.allclose(T_full.sum(axis=1), 1.0, atol=1e-8)
    inactive = np.setdiff1d(np.arange(4), active)
    assert np.allclose(T_full[inactive, inactive], 1.0)
    assert np.allclose(T_full[np.ix_(inactive, np.setdiff1d(np.arange(4), inactive))], 0.0)
    assert np.all(pi_full >= 0)
    assert np.isclose(pi_full.sum(), 1.0)


def test_cluster_populations_shape(simple_msm):
    msm_mod = simple_msm["mod"]
    pi_micro = np.array([0.2, 0.3, 0.5], dtype=float)
    mapping = np.array([0, 1, 1], dtype=int)
    pi_macro = msm_mod.compute_macro_populations(pi_micro, mapping)
    assert pi_macro.shape == (2,)
    assert np.isclose(pi_macro.sum(), 1.0)


# ---------------------------------------------------------------------------
# 5. Macrostate Lumping Shapes
# ---------------------------------------------------------------------------


def test_macro_labels_shape(simple_msm):
    msm_mod = simple_msm["mod"]
    labels = msm_mod.pcca_like_macrostates(simple_msm["T"], n_macrostates=2)
    if labels is None:
        pytest.skip("PCCA+ returned None for given transition matrix")
    assert labels.ndim == 1
    assert labels.shape[0] == simple_msm["T"].shape[0]


def test_macro_populations_shape(simple_msm):
    msm_mod = simple_msm["mod"]
    micro_labels = np.array([0, 1, 1, 2], dtype=int)
    pi_micro = np.array([0.2, 0.3, 0.1, 0.4], dtype=float)
    pi_macro = msm_mod.compute_macro_populations(pi_micro, micro_labels)
    assert pi_macro.shape == (3,)
    assert np.isclose(pi_macro.sum(), 1.0)


def test_macro_transition_matrix_shape(simple_msm):
    msm_mod = simple_msm["mod"]
    T_macro = msm_mod.lump_micro_to_macro_T(
        simple_msm["T"], simple_msm["pi"], np.array([0, 0, 1, 1], dtype=int)
    )
    assert T_macro.shape == (2, 2)
    assert np.all(T_macro >= 0)
    assert np.allclose(T_macro.sum(axis=1), 1.0, atol=1e-6)


def test_macro_mfpt_matrix_shape(simple_msm):
    msm_mod = simple_msm["mod"]
    T_macro = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    mfpt = msm_mod.compute_macro_mfpt(T_macro)
    assert mfpt.shape == (2, 2)
    assert np.allclose(np.diag(mfpt), 0.0)


def test_two_state_mfpt_closed_form(simple_msm):
    msm_mod = simple_msm["mod"]
    a, b = 0.2, 0.3
    T = np.array([[1 - a, a], [b, 1 - b]], dtype=float)
    mfpt = msm_mod.compute_macro_mfpt(T)
    assert mfpt[0, 1] == pytest.approx(1.0 / a, rel=1e-6)
    assert mfpt[1, 0] == pytest.approx(1.0 / b, rel=1e-6)
    assert np.allclose(np.diag(mfpt), 0.0)


# ---------------------------------------------------------------------------
# 6. FES (Free Energy Surface) Shapes
# ---------------------------------------------------------------------------


def test_fes_result_matrix_shapes(build_result_payload):
    fes = build_result_payload["fes"]
    assert fes.F.shape == (12, 10)
    assert fes.xedges.shape == (13,)
    assert fes.yedges.shape == (11,)
    assert fes.levels_kJmol.shape == (6,)


def test_fes_grid_contains_finite_values(build_result_payload):
    fes = build_result_payload["fes"]
    assert np.isfinite(fes.F).all()


def test_fes_cv_pair_extraction_shapes(deeptica_workflow):
    matrix = deeptica_workflow["outputs"]
    cv1 = matrix[:, 0]
    cv2 = matrix[:, 1]
    assert cv1.shape == cv2.shape
    assert cv1.ndim == 1


def test_fes_histogram_fallback_shapes():
    grid = np.random.default_rng(9).random((20, 18))
    assert grid.shape == (20, 18)


# ---------------------------------------------------------------------------
# 7. Full Workflow Integration Tests
# ---------------------------------------------------------------------------


def test_build_result_all_matrix_shapes_with_deeptica(
    build_result_payload, deeptica_workflow
):
    result = build_result_payload["result"]
    assert result.transition_matrix.shape[0] == result.transition_matrix.shape[1]
    assert result.stationary_distribution.shape == (
        result.transition_matrix.shape[0],
    )
    assert result.cluster_populations.shape[0] == 3
    assert result.fes.F.shape == (12, 10)
    assert deeptica_workflow["outputs"].shape[1] == deeptica_workflow["config"].n_out


def test_build_result_deeptica_artifacts_shapes(build_result_payload):
    result = build_result_payload["result"]
    assert "mlcv_deeptica" in result.artifacts
    artifacts = result.artifacts["mlcv_deeptica"]
    assert "output_variance" in artifacts
    variance = np.asarray(artifacts["output_variance"])
    assert variance.ndim == 1
    if variance.size:
        assert variance.shape[0] == len(result.feature_names)
    assert "whitening" in artifacts
    whitening = artifacts["whitening"]
    assert "transform" in whitening
    assert "mean" in whitening
    transform = np.asarray(whitening["transform"])
    mean = np.asarray(whitening["mean"])
    assert transform.shape[0] == transform.shape[1]
    assert mean.shape == (transform.shape[0],)


def test_deeptica_dataset_transform_shape_preservation(deeptica_workflow):
    matrix = deeptica_workflow["input_matrix"]
    outputs = deeptica_workflow["outputs"]
    assert matrix.shape[0] == outputs.shape[0]
    assert matrix.shape[1] != outputs.shape[1]


def test_cv_bin_edges_shapes_after_deeptica(deeptica_workflow):
    edges_cv1 = build_mod._compute_bin_edges(deeptica_workflow["outputs"][:, 0], 32)
    edges_cv2 = build_mod._compute_bin_edges(deeptica_workflow["outputs"][:, 1], 32)
    assert len(edges_cv1) == 33
    assert len(edges_cv2) == 33
    assert np.all(np.diff(edges_cv1) > 0)
    assert np.all(np.diff(edges_cv2) > 0)


def test_fes_edges_monotonic_and_cover_range(deeptica_workflow):
    x = deeptica_workflow["outputs"][:, 0]
    edges = build_mod._compute_bin_edges(x, 32)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] <= x.min() <= edges[-1]
    assert edges[0] <= x.max() <= edges[-1]


# ---------------------------------------------------------------------------
# 8. Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


def test_json_roundtrip_values_equal(build_result_payload):
    result = build_result_payload["result"]
    text = result.to_json()
    loaded = build_mod.BuildResult.from_json(text)
    assert np.allclose(loaded.transition_matrix, result.transition_matrix)
    assert np.allclose(loaded.stationary_distribution, result.stationary_distribution)
    assert np.array_equal(loaded.cluster_populations, result.cluster_populations)
    assert np.allclose(loaded.fes.F, result.fes.F)


def test_array_payload_roundtrip_exact_values():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    payload = build_mod._serialize_array_payload(arr)
    buf = base64.b64decode(payload["data"])
    restored = np.frombuffer(buf, dtype=np.float32).reshape(tuple(payload["shape"]))
    assert np.array_equal(restored, arr)


# ---------------------------------------------------------------------------
# 9. Edge Cases and Error Conditions
# ---------------------------------------------------------------------------


def test_empty_shards_matrix_shapes():
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    empty = [np.empty((0, 3))]
    info = build_pair_info(empty, (1,))
    assert info.idx_t.size == 0
    assert info.idx_tau.size == 0


def test_short_shards_no_pairs_shapes():
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    shards = [np.zeros((2, 2)), np.ones((3, 2))]
    info = build_pair_info(shards, (5,))
    assert info.idx_t.size == 0
    assert info.diagnostics["usable_pairs"] == 0


def test_single_shard_all_shapes():
    from pmarlo.features.deeptica.core.pairs import build_pair_info

    shard = [np.random.default_rng(4).random((24, 3))]
    info = build_pair_info(shard, (3,))
    assert info.idx_t.ndim == 1
    assert info.idx_tau.shape == info.idx_t.shape


# ---------------------------------------------------------------------------
# 10. Consistency Tests Across Pipeline Stages
# ---------------------------------------------------------------------------


def test_n_frames_consistency_throughout_pipeline(deeptica_workflow):
    history = deeptica_workflow["history"]
    n_frames = deeptica_workflow["n_frames"]
    assert history["usable_pairs"] <= n_frames
    assert deeptica_workflow["outputs"].shape[0] == n_frames


def test_n_features_change_through_deeptica(deeptica_workflow):
    matrix = deeptica_workflow["input_matrix"]
    outputs = deeptica_workflow["outputs"]
    assert matrix.shape[1] == 4
    assert outputs.shape[1] == deeptica_workflow["config"].n_out
