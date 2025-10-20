from __future__ import annotations

import pytest
import yaml

from pmarlo.settings import ConfigurationError, load_defaults


def test_load_defaults_requires_keys(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("enable_cv_bias: true
", encoding="utf-8")
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()
    with pytest.raises(ConfigurationError):
        load_defaults()
    load_defaults.cache_clear()


def test_load_defaults_validates_bias_mode(monkeypatch, tmp_path):
    cfg = {
        "enable_cv_bias": True,
        "bias_mode": "invalid",
        "torch_threads": 2,
        "precision": "single",
        "feature_spec_path": "feature_spec.yaml",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()
    with pytest.raises(ConfigurationError):
        load_defaults()
    load_defaults.cache_clear()
