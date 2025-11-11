from __future__ import annotations

import pytest
import yaml

from pmarlo.settings import ConfigurationError, load_defaults


def test_load_defaults_requires_keys(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("enable_cv_bias: true\n", encoding="utf-8")
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


def test_load_defaults_accepts_string_boolean(monkeypatch, tmp_path):
    cfg = {
        "enable_cv_bias": "false",
        "bias_mode": "harmonic",
        "torch_threads": 2,
        "precision": "single",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()
    cfg_out = load_defaults()
    load_defaults.cache_clear()
    assert cfg_out["enable_cv_bias"] is False


def test_load_defaults_rejects_invalid_boolean(monkeypatch, tmp_path):
    cfg = {
        "enable_cv_bias": "definitely",
        "bias_mode": "harmonic",
        "torch_threads": 2,
        "precision": "single",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()
    with pytest.raises(ConfigurationError):
        load_defaults()
    load_defaults.cache_clear()


def test_load_defaults_expands_user_in_feature_spec_path(monkeypatch, tmp_path):
    cfg = {
        "enable_cv_bias": True,
        "bias_mode": "harmonic",
        "torch_threads": 2,
        "precision": "single",
        "feature_spec_path": "~/spec.yaml",
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("features: []\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(cfg_path))
    load_defaults.cache_clear()
    cfg_out = load_defaults()
    load_defaults.cache_clear()
    assert cfg_out["feature_spec_path"] == str(spec_path)


def test_load_defaults_reloads_when_env_changes(monkeypatch, tmp_path):
    first_cfg = {
        "enable_cv_bias": True,
        "bias_mode": "harmonic",
        "torch_threads": 2,
        "precision": "single",
    }
    second_cfg = dict(first_cfg)
    second_cfg["torch_threads"] = 8

    first_path = tmp_path / "first.yaml"
    first_path.write_text(yaml.safe_dump(first_cfg), encoding="utf-8")
    second_path = tmp_path / "second.yaml"
    second_path.write_text(yaml.safe_dump(second_cfg), encoding="utf-8")

    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(first_path))
    load_defaults.cache_clear()
    first_loaded = load_defaults()
    assert first_loaded["torch_threads"] == 2

    monkeypatch.setenv("PMARLO_CONFIG_FILE", str(second_path))
    second_loaded = load_defaults()
    load_defaults.cache_clear()
    assert second_loaded["torch_threads"] == 8
