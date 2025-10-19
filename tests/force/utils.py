from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from openmm import unit
from openmm.app import PDBFile

from pmarlo.features.deeptica.cv_bias_potential import CVBiasPotential
from pmarlo.features.deeptica.export import export_cv_bias_potential
from pmarlo.features.deeptica.ts_feature_extractor import (
    build_feature_extractor_module,
    canonicalize_feature_spec,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "data" / "feature_spec.yaml"
PDB_PATH = Path(__file__).resolve().parents[1] / "data" / "ala2.pdb"
BOX_LENGTHS = torch.tensor([1.5, 1.5, 1.5], dtype=torch.float32)


def load_spec_dict() -> dict:
    with SPEC_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def canonical_spec():
    return canonicalize_feature_spec(load_spec_dict())


def reference_positions() -> np.ndarray:
    pdb = PDBFile(str(PDB_PATH))
    pos = pdb.positions.value_in_unit(unit.nanometer)
    return np.array(pos, dtype=np.float64)


def default_box_tensor() -> torch.Tensor:
    return torch.diag(BOX_LENGTHS.clone())


def _build_network(input_dim: int) -> torch.nn.Module:
    torch.manual_seed(1234)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )


class IdentityScaler:
    def __init__(self, size: int) -> None:
        self.mean_ = np.zeros(size, dtype=np.float32)
        self.scale_ = np.ones(size, dtype=np.float32)


def build_eager_module() -> Tuple[CVBiasPotential, dict]:
    spec_dict = load_spec_dict()
    canonical = canonicalize_feature_spec(spec_dict)
    extractor = build_feature_extractor_module(canonical)
    network = _build_network(canonical.n_features)
    scaler = IdentityScaler(canonical.n_features)
    spec_hash = hashlib.sha256(
        json.dumps(spec_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    module = CVBiasPotential(
        cv_model=network,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_extractor=extractor,
        feature_spec_hash=spec_hash,
        bias_strength=5.0,
        feature_names=list(canonical.feature_names),
    )
    return module, spec_dict


def export_bias_module(tmpdir: Path, model_name: str = "bias_fixture"):
    module, spec_dict = build_eager_module()
    canonical = canonicalize_feature_spec(spec_dict)
    scaler = IdentityScaler(canonical.n_features)
    bundle = export_cv_bias_potential(
        network=module.cv_model,
        scaler=scaler,
        history={"n_out": 2},
        output_dir=tmpdir,
        feature_spec=spec_dict,
        model_name=model_name,
        bias_strength=5.0,
        feature_names=list(canonical.feature_names),
    )
    return module, bundle
