import math

import torch

from pmarlo.features.deeptica.cv_bias_potential import (
    CVBiasPotential,
    HarmonicExpansionBias,
)
from pmarlo.features.deeptica.ts_feature_extractor import (
    build_feature_extractor_module,
    canonicalize_feature_spec,
)


def _sample_spec():
    return {
        "use_pbc": False,
        "features": [
            {"name": "d01", "type": "distance", "atoms": [0, 1]},
            {"name": "angle012", "type": "angle", "atoms": [0, 1, 2]},
            {"name": "torsion0123", "type": "dihedral", "atoms": [0, 1, 2, 3]},
        ],
    }


def test_feature_extractor_forward_and_gradients():
    spec = canonicalize_feature_spec(_sample_spec())
    extractor = build_feature_extractor_module(spec)
    extractor.eval()

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    box = torch.eye(3, dtype=torch.float32)

    features = extractor(positions, box)
    assert features.shape == (3,)
    assert torch.allclose(features[0], torch.tensor(1.0))
    assert torch.allclose(features[1], torch.tensor(math.pi / 2), atol=1e-6)
    assert torch.allclose(features[2], torch.tensor(math.pi / 2), atol=1e-6)

    total = features.sum()
    total.backward()
    assert positions.grad is not None
    assert torch.all(torch.isfinite(positions.grad))

    scripted = torch.jit.script(extractor)
    scripted_features = scripted(positions.detach(), box)
    assert torch.allclose(scripted_features, features.detach())


def test_cvbiaspotential_composes_feature_extractor():
    spec = canonicalize_feature_spec(_sample_spec())
    extractor = build_feature_extractor_module(spec)
    scaler_mean = torch.zeros(len(spec.feature_names), dtype=torch.float32)
    scaler_scale = torch.ones(len(spec.feature_names), dtype=torch.float32)

    cv_model = torch.nn.Linear(len(spec.feature_names), 1, bias=False)
    cv_model.weight.data.fill_(1.0)
    module = CVBiasPotential(
        cv_model=cv_model,
        scaler_mean=scaler_mean.numpy(),
        scaler_scale=scaler_scale.numpy(),
        feature_extractor=extractor,
        feature_spec_hash="dummyhash",
        bias_strength=2.0,
        feature_names=list(spec.feature_names),
    )
    module.eval()

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    box = torch.eye(3, dtype=torch.float32)
    energy = module(positions, box)
    assert energy.shape == torch.Size([])
    energy.backward()
    assert positions.grad is not None
    assert torch.all(torch.isfinite(positions.grad))

    scripted = torch.jit.script(module)
    scripted_energy = scripted(positions.detach(), box)
    assert torch.allclose(scripted_energy, energy.detach())
