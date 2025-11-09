"""Test model file discovery for CV export."""

import pytest
from pathlib import Path


def test_model_timestamp_extraction():
    """Test extracting timestamp from checkpoint and model filenames."""
    checkpoint_dir_name = "training-20251108-193156"
    model_file_name = "deeptica-20251108-195911.pt"

    checkpoint_timestamp = checkpoint_dir_name.replace("training-", "")
    model_timestamp = model_file_name.replace("deeptica-", "").replace(".pt", "")

    assert checkpoint_timestamp == "20251108-193156"
    assert model_timestamp == "20251108-195911"


def test_model_timestamp_comparison():
    """Test that model timestamp comparison works correctly."""
    checkpoint_timestamp = "20251108-193156"

    # Model saved after checkpoint starts
    model_timestamp_after = "20251108-195911"
    assert model_timestamp_after >= checkpoint_timestamp

    # Model saved before checkpoint starts
    model_timestamp_before = "20251108-173316"
    assert not (model_timestamp_before >= checkpoint_timestamp)


def test_filter_scaler_files():
    """Test that .scaler.pt files are filtered out."""
    model_files = [
        "deeptica-20251108-195911.pt",
        "deeptica-20251108-195911.scaler.pt",
        "deeptica-20251108-173316.pt",
        "deeptica-20251108-173316.scaler.pt",
    ]

    filtered = [f for f in model_files if not f.endswith(".scaler.pt")]

    assert len(filtered) == 2
    assert "deeptica-20251108-195911.pt" in filtered
    assert "deeptica-20251108-173316.pt" in filtered
    assert "deeptica-20251108-195911.scaler.pt" not in filtered


def test_model_companion_files():
    """Test that companion files are identified correctly."""
    model_file = "deeptica-20251108-195911"

    expected_files = {
        "pt": f"{model_file}.pt",
        "scaler": f"{model_file}.scaler.pt",
        "json": f"{model_file}.json",
    }

    assert expected_files["pt"] == "deeptica-20251108-195911.pt"
    assert expected_files["scaler"] == "deeptica-20251108-195911.scaler.pt"
    assert expected_files["json"] == "deeptica-20251108-195911.json"


def test_model_selection_logic():
    """Test the complete model selection logic."""
    checkpoint_timestamp = "20251108-193156"

    # Available models
    available_models = [
        "deeptica-20251108-173316.pt",
        "deeptica-20251108-195911.pt",
        "deeptica-20251109-120000.pt",
    ]

    # Filter and find matching models
    matching_models = []
    for model in available_models:
        model_timestamp = model.replace("deeptica-", "").replace(".pt", "")
        if model_timestamp >= checkpoint_timestamp:
            matching_models.append(model)

    # Should match models saved at or after checkpoint time
    assert len(matching_models) == 2
    assert "deeptica-20251108-195911.pt" in matching_models
    assert "deeptica-20251109-120000.pt" in matching_models

    # Should select the first (earliest) matching model
    selected_model = matching_models[0]
    assert selected_model == "deeptica-20251108-195911.pt"
