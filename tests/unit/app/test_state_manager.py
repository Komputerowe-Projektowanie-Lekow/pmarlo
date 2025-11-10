from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pmarlo_webapp.app.backend.state import StateManager


def _fake_layout(tmp_path: Path):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    models.mkdir(parents=True, exist_ok=True)
    state_path = workspace / "state.json"
    return SimpleNamespace(
        workspace_dir=workspace,
        models_dir=models,
        state_path=state_path,
    )


def test_state_manager_discovers_missing_models(tmp_path):
    layout = _fake_layout(tmp_path)
    checkpoint_dir = layout.models_dir / "training-20250101-010101"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = checkpoint_dir / "deeptica_cv_model_metadata.json"
    metadata_path.write_text(
        json.dumps({"cv_dim": 2, "feature_spec_sha256": "abc123"}),
        encoding="utf-8",
    )
    (checkpoint_dir / "deeptica_cv_model.pt").write_text("", encoding="utf-8")
    (checkpoint_dir / "deeptica_cv_model_scaler.npz").write_bytes(b"dummy")
    (checkpoint_dir / "deeptica_cv_model_config.json").write_text(
        "{}", encoding="utf-8"
    )

    bundle_payload = {
        "metadata": {
            "dataset_hash": "hash-123",
            "seed": 42,
            "temperature": 305.0,
            "applied_opts": {
                "bins": {"Rg": 64},
                "actual_plan": {
                    "steps": [
                        {
                            "name": "LEARN_CV",
                            "params": {
                                "lag": 75,
                                "hidden": [64, 64],
                                "max_epochs": 100,
                                "early_stopping": 15,
                                "tau_schedule": [2, 5, 10],
                                "val_tau": 10,
                                "epochs_per_tau": 12,
                                "gradient_clip_val": 0.5,
                                "learning_rate": 1e-4,
                                "weight_decay": 0.0,
                                "checkpoint_dir": str(checkpoint_dir),
                            },
                        }
                    ]
                },
            },
        },
        "artifacts": {
            "mlcv_deeptica": {
                "val_loss": [1.0, 0.8],
            }
        },
    }
    bundle_path = layout.models_dir / "deeptica-20250101-010101.pbz"
    bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")

    state = StateManager(layout.state_path, workspace_layout=layout)
    models = state.models

    assert len(models) == 1

    entry = models[0]
    assert entry["bundle"] == str(bundle_path.resolve())
    assert entry["lag"] == 75
    assert entry["dataset_hash"] == "hash-123"
    assert entry["bins"] == {"Rg": 64}
    assert entry["hidden"] == [64, 64]
    assert entry["tau_schedule"] == [2, 5, 10]
    assert entry["cv_model_bundle"]["cv_dim"] == 2
    assert entry["cv_model_bundle"]["feature_spec_sha256"] == "abc123"

    # Update entry to ensure cv metadata persists
    updated = dict(entry)
    updated["cv_model_bundle"] = {"model_path": "C:/tmp/deeptica_cv_model.pt"}
    state.update_model(0, updated)
    assert state.models[0]["cv_model_bundle"]["model_path"].endswith(
        "deeptica_cv_model.pt"
    )
