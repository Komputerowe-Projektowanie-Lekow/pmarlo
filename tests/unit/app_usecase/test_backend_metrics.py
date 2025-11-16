import pytest

from pmarlo.api import normalize_training_metrics


def test_normalize_training_metrics_requires_best_fields():
    metrics = {"val_score_curve": [0.1, 0.4, 0.6, 0.95]}
    with pytest.raises(ValueError):
        normalize_training_metrics(metrics, tau_schedule=[2, 5], epochs_per_tau=2)


def test_normalize_training_metrics_validates_and_coerces_values():
    metrics = {
        "best_val_score": "0.3",
        "best_epoch": 2.0,
        "best_tau": "7",
    }
    normalized = normalize_training_metrics(metrics)
    assert normalized["best_val_score"] == pytest.approx(0.3)
    assert normalized["best_epoch"] == 2
    assert normalized["best_tau"] == 7
