from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from openmm.app import ForceField, PDBFile

from pmarlo.settings import load_defaults, resolve_feature_spec_path
from pmarlo.replica_exchange.system_builder import create_system
from .utils import export_bias_module, PDB_PATH


@pytest.fixture()
def model_bundle(tmp_path):
    module, bundle = export_bias_module(tmp_path)
    return module, bundle


@pytest.fixture()
def pdb_forcefield():
    pdb = PDBFile(str(PDB_PATH))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    return pdb, forcefield


def _write_config(tmp_path: Path, spec_path: Path, enable_bias: bool = True) -> Path:
    config = {
        "enable_cv_bias": enable_bias,
        "bias_mode": "harmonic",
        "torch_threads": 2,
        "precision": "single",
        "feature_spec_path": str(spec_path),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return cfg_path


def test_create_system_accepts_matching_spec(monkeypatch, tmp_path, model_bundle, pdb_forcefield):
    _, bundle = model_bundle
    spec_source = resolve_feature_spec_path()
    spec_copy = tmp_path / "feature_spec.yaml"
    spec_copy.write_text(spec_source.read_text(encoding="utf-8"), encoding="utf-8")
    cfg_path = _write_config(tmp_path, spec_copy)
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()

    pdb, forcefield = pdb_forcefield
    create_system(pdb, forcefield, cv_model_path=str(bundle.model_path))
    load_defaults.cache_clear()



def test_create_system_rejects_mismatched_spec(monkeypatch, tmp_path, model_bundle, pdb_forcefield):
    _, bundle = model_bundle
    spec_source = resolve_feature_spec_path()
    spec_payload = yaml.safe_load(spec_source.read_text(encoding="utf-8"))
    assert isinstance(spec_payload, dict)
    features = list(spec_payload.get("features", []))
    if not features:
        pytest.skip("Feature spec unexpectedly empty")
    features.pop()
    spec_payload["features"] = features
    mismatched_spec = tmp_path / "feature_spec_bad.yaml"
    mismatched_spec.write_text(yaml.safe_dump(spec_payload), encoding="utf-8")
    cfg_path = _write_config(tmp_path, mismatched_spec)
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()

    pdb, forcefield = pdb_forcefield
    with pytest.raises(RuntimeError, match="Feature specification mismatch"):
        create_system(pdb, forcefield, cv_model_path=str(bundle.model_path))
    load_defaults.cache_clear()

    load_defaults.cache_clear()

