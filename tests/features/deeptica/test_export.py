import hashlib
import json

import numpy as np
import torch

from pmarlo.features.deeptica.export import (
    export_cv_bias_potential,
    load_cv_model_info,
)


class DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)


def _spec():
    return {
        "use_pbc": False,
        "features": [
            {"name": "d01", "type": "distance", "atoms": [0, 1]},
            {"name": "a012", "type": "angle", "atoms": [0, 1, 2]},
            {"name": "t0123", "type": "dihedral", "atoms": [0, 1, 2, 3]},
        ],
    }


def test_export_bias_pipeline(tmp_path):
    feature_spec = _spec()
    n_features = len(feature_spec["features"])
    network = torch.nn.Linear(n_features, 2)
    scaler = DummyScaler(mean=np.zeros(n_features), scale=np.ones(n_features))
    history = {"n_out": 2}

    bundle = export_cv_bias_potential(
        network=network,
        scaler=scaler,
        history=history,
        output_dir=tmp_path,
        feature_spec=feature_spec,
        model_name="test_bias",
        bias_strength=5.0,
    )

    assert bundle.model_path.exists()
    assert bundle.scaler_path.exists()
    assert bundle.metadata_path.exists()

    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(
        json.dumps(feature_spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert metadata["feature_spec_sha256"] == expected_hash
    assert bundle.feature_spec_hash == expected_hash

    loaded = torch.jit.load(str(bundle.model_path), map_location="cpu")
    assert getattr(loaded, "feature_spec_sha256") == expected_hash

    info = load_cv_model_info(tmp_path, model_name="test_bias")
    assert info["feature_spec_sha256"] == expected_hash
