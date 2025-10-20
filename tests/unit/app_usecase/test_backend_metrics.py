from example_programs.app_usecase.app.backend import _normalize_training_metrics


def test_normalize_training_metrics_infers_best_values():
    metrics = {
        "val_score_curve": [0.1, 0.4, 0.6, 0.95],
    }
    normalized = _normalize_training_metrics(
        metrics,
        tau_schedule=[2, 5],
        epochs_per_tau=2,
    )
    assert normalized["best_val_score"] == 0.95
    assert normalized["best_epoch"] == 4
    assert normalized["best_tau"] == 5


def test_normalize_training_metrics_preserves_existing_best_values():
    metrics = {
        "val_score_curve": [0.2, 0.3, 0.25],
        "best_val_score": 0.3,
        "best_epoch": 2,
        "best_tau": 7,
    }
    normalized = _normalize_training_metrics(metrics, tau_schedule=[2, 7], epochs_per_tau=2)
    assert normalized["best_val_score"] == 0.3
    assert normalized["best_epoch"] == 2
    assert normalized["best_tau"] == 7
