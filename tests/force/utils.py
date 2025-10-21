from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from openmm import unit
from openmm.app import PDBFile

from pmarlo.features.deeptica.cv_bias_potential import CVBiasPotential
from pmarlo.features.deeptica.export import export_cv_bias_potential
from pmarlo.features.deeptica.ts_feature_extractor import (
    build_feature_extractor_module,
    canonicalize_feature_spec,
)
from pmarlo.settings import load_feature_spec

PDB_PATH = Path(__file__).resolve().parents[1] / "_assets" / "3gd8-fixed.pdb"


def _compute_box_lengths() -> torch.Tensor:
    pdb = PDBFile(str(PDB_PATH))
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise RuntimeError("3gd8-fixed.pdb must define periodic box vectors.")
    lengths: list[float] = []
    for vec in vectors:
        components = vec.value_in_unit(unit.nanometer)
        lengths.append(
            float(
                np.linalg.norm(
                    np.asarray([components[0], components[1], components[2]])
                )
            )
        )
    return torch.tensor(lengths, dtype=torch.float32)


BOX_LENGTHS = _compute_box_lengths()


def load_spec_dict() -> dict:
    spec, _ = load_feature_spec()
    return spec


def spec_hash() -> str:
    _, hash_value = load_feature_spec()
    return hash_value


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
    module = CVBiasPotential(
        cv_model=network,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_extractor=extractor,
        feature_spec_hash=spec_hash(),
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
