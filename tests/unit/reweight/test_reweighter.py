from __future__ import annotations

import numpy as np
import pytest

from pmarlo.reweight.reweighter import Reweighter


def make_dataset(
    energy: np.ndarray | None = None,
    *,
    bias: np.ndarray | None = None,
    base: np.ndarray | None = None,
):
    split = {
        "beta": 1.0 / (0.00831446261815324 * 310.0),  # simulation temperature 310K
    }
    if energy is not None:
        split["energy"] = energy
    if bias is not None:
        split["bias"] = bias
    if base is not None:
        split["w_frame"] = base
    return {"splits": {"s1": split}}


def test_reweighter_produces_w_frame_only():
    energy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ds = make_dataset(energy)
    rw = Reweighter(temperature_ref_K=300.0)
    out = rw.apply(ds)
    assert "s1" in out
    # Canonical key present
    assert "w_frame" in ds["splits"]["s1"]
    np.testing.assert_allclose(out["s1"], ds["splits"]["s1"]["w_frame"], rtol=0, atol=0)
    np.testing.assert_allclose(out["s1"].sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_reweighter_rejects_old_weights_alias():
    energy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ds = make_dataset(energy)
    ds["splits"]["s1"]["weights"] = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    rw = Reweighter(temperature_ref_K=300.0)

    with pytest.raises(ValueError) as excinfo:
        rw.apply(ds)

    assert "deprecated key 'weights'" in str(excinfo.value)


def test_reweighter_uses_existing_base_weights():
    # If energy is constant, unnormalized weights would be uniform; base weights shape outcome
    energy = np.ones(
        4, dtype=np.float64
    )  # constant energy => base weighting effect visible
    base = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    ds = make_dataset(energy, base=base)
    rw = Reweighter(temperature_ref_K=300.0)
    out = rw.apply(ds)["s1"]
    # Expected final weights proportional to base (normalized)
    expected = base / base.sum()
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_reweighter_missing_energy_raises():
    ds = make_dataset(None)
    rw = Reweighter(temperature_ref_K=300.0)
    with pytest.raises(ValueError) as ei:
        rw.apply(ds)
    assert "missing required 'energy'" in str(ei.value)


def test_reweighter_non_positive_sum_raises():
    energy = np.zeros(3, dtype=np.float64)  # uniform base factor from energy
    base = np.zeros(3, dtype=np.float64)  # force zero normalization sum
    ds = make_dataset(energy, base=base)
    rw = Reweighter(temperature_ref_K=300.0)
    with pytest.raises(ValueError) as ei:
        rw.apply(ds)
    assert "non-finite or non-positive" in str(ei.value)


def test_reweighter_deterministic_cache():
    energy = np.array([0.5, 0.1, 0.3], dtype=np.float64)
    ds = make_dataset(energy)
    rw = Reweighter(temperature_ref_K=300.0)
    first = rw.apply(ds)["s1"].copy()
    # Modify nothing; call again should return identical weights (byte-for-byte)
    second = rw.apply(ds)["s1"]
    assert first.shape == second.shape
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_bias_changes_relative_weights():
    # Without bias: lower energy -> higher weight. Add a bias that inverts ordering.
    energy = np.array([1.0, 2.0], dtype=np.float64)
    # Bias penalize first frame strongly
    bias = np.array([10.0, 0.0], dtype=np.float64)
    ds = make_dataset(energy, bias=bias)
    rw = Reweighter(temperature_ref_K=300.0)
    w = rw.apply(ds)["s1"]
    # Second frame should now dominate
    assert w[1] > w[0]
    np.testing.assert_allclose(w.sum(), 1.0)


def test_bias_length_mismatch_raises():
    energy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bias = np.array([0.0, 0.1], dtype=np.float64)
    ds = make_dataset(energy, bias=bias)
    rw = Reweighter(temperature_ref_K=300.0)

    with pytest.raises(ValueError) as excinfo:
        rw.apply(ds)

    assert "bias length mismatch" in str(excinfo.value)


def test_inputs_remain_immutable_after_apply():
    energy = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    bias = np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float64)
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    energy_before = energy.copy()
    bias_before = bias.copy()
    base_before = base.copy()

    ds = make_dataset(energy, bias=bias, base=base)
    rw = Reweighter(temperature_ref_K=310.0)
    rw.apply(ds)

    np.testing.assert_allclose(energy, energy_before, rtol=0, atol=0)
    np.testing.assert_allclose(bias, bias_before, rtol=0, atol=0)
    np.testing.assert_allclose(base, base_before, rtol=0, atol=0)


def _build_tram_dataset() -> dict[str, object]:
    k_B = 0.00831446261815324
    temps = [300.0, 320.0]
    betas = [1.0 / (k_B * T) for T in temps]
    energies = [
        np.array([1.0, 0.8, 1.2, 0.7], dtype=np.float64),
        np.array([0.9, 1.1, 0.85, 1.25], dtype=np.float64),
    ]
    dtrajs = [
        np.array([0, 1, 0, 1], dtype=np.int32),
        np.array([1, 0, 1, 0], dtype=np.int32),
    ]
    bias_matrices = [
        np.array([[0.0, 0.2], [0.1, 0.3], [0.0, 0.2], [0.1, 0.3]], dtype=np.float64),
        np.array([[0.2, 0.0], [0.3, 0.1], [0.2, 0.0], [0.3, 0.1]], dtype=np.float64),
    ]

    splits: dict[str, dict[str, object]] = {}
    for idx, (energy, beta) in enumerate(zip(energies, betas)):
        split = {
            "beta": beta,
            "energy": energy,
            "tram": {"trajectory_index": idx, "therm_state_index": idx},
        }
        splits[f"s{idx}"] = split

    dataset: dict[str, object] = {
        "splits": splits,
        "tram_dataset": {
            "dtrajs": dtrajs,
            "bias_matrices": bias_matrices,
        },
    }
    return dataset


def test_tram_mode_matches_deeptime_weights():
    pytest.importorskip("deeptime")
    from deeptime.markov.msm import TRAM, TRAMDataset

    dataset = _build_tram_dataset()
    rw = Reweighter(temperature_ref_K=300.0)
    weights = rw.apply(dataset, mode="TRAM")

    tram_inputs = dataset["tram_dataset"]
    assert isinstance(tram_inputs, dict)
    tram_dataset = TRAMDataset(
        dtrajs=tram_inputs["dtrajs"],
        bias_matrices=tram_inputs["bias_matrices"],
    )
    tram_model = TRAM(lagtime=1, init_strategy="MBAR").fit(tram_dataset).fetch_model()

    log_weights_state0 = tram_model.compute_sample_weights_log(
        tram_inputs["dtrajs"], tram_inputs["bias_matrices"], therm_state=0
    )
    log_weights_state1 = tram_model.compute_sample_weights_log(
        tram_inputs["dtrajs"], tram_inputs["bias_matrices"], therm_state=1
    )

    expected0 = np.exp(log_weights_state0[0] - np.max(log_weights_state0[0]))
    expected0 /= expected0.sum()
    expected1 = np.exp(log_weights_state1[1] - np.max(log_weights_state1[1]))
    expected1 /= expected1.sum()

    np.testing.assert_allclose(weights["s0"], expected0)
    np.testing.assert_allclose(weights["s1"], expected1)


def test_tram_mode_requires_dataset_payload():
    energy = np.array([0.5, 0.1, 0.2], dtype=np.float64)
    ds = make_dataset(energy)
    rw = Reweighter(temperature_ref_K=290.0)

    with pytest.raises(ValueError):
        rw.apply(ds, mode="TRAM")


def test_tram_bias_matrix_state_mismatch_raises():
    energy = np.array([1.0, 1.1, 0.9], dtype=np.float64)
    ds = make_dataset(energy)
    splits = ds["splits"]
    split = splits["s1"]
    split["tram"] = {"trajectory_index": 0, "therm_state_index": 0}

    tram_dataset = {
        "dtrajs": [np.array([0, 1, 0], dtype=np.int32)],
        "bias_matrices": [
            np.array(
                [[0.0, 0.1], [0.2, 0.3], [0.1, 0.0]],
                dtype=np.float64,
            ),
        ],
    }

    # Append a second bias matrix with a different thermodynamic-state count
    tram_dataset["dtrajs"].append(np.array([1, 0, 1], dtype=np.int32))
    tram_dataset["bias_matrices"].append(
        np.array(
            [[0.0, 0.1, 0.2], [0.1, 0.0, 0.1], [0.2, 0.1, 0.0]],
            dtype=np.float64,
        )
    )

    ds["tram_dataset"] = tram_dataset

    rw = Reweighter(temperature_ref_K=300.0)

    with pytest.raises(ValueError) as excinfo:
        rw.apply(ds, mode="TRAM")

    assert "bias matrices must all share" in str(excinfo.value)
