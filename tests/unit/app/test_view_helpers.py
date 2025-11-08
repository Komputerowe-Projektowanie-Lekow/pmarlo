import importlib.util
from pathlib import Path
import sys


def _load_view_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "pmarlo_webapp" / "app" / "core" / "view_helpers.py"
    app_root = repo_root / "pmarlo_webapp" / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    spec = importlib.util.spec_from_file_location(
        "pmarlo_webapp.app.core.view_helpers_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


view_helpers = _load_view_helpers()


def test_selector_stats_counts_selected() -> None:
    records = [
        {"run_id": "run-1", "n_shards": 2, "frames_total": 100},
        {"run_id": "run-2", "n_shards": 3, "frames_total": 250},
    ]

    stats = view_helpers._aggregate_shard_selector_stats(records, ["run-1"])

    assert stats["runs_total"] == 2
    assert stats["runs_selected"] == 1
    assert stats["shards_total"] == 5
    assert stats["shards_selected"] == 2
    assert stats["frames_total"] == 350
    assert stats["frames_selected"] == 100


def test_selector_stats_handles_empty_selection() -> None:
    records = [
        {"run_id": "run-1", "n_shards": 1, "frames_total": 10},
    ]

    stats = view_helpers._aggregate_shard_selector_stats(records, [])

    assert stats["runs_selected"] == 0
    assert stats["shards_selected"] == 0
    assert stats["frames_selected"] == 0
