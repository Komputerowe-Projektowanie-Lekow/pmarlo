from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_deeptica_trainer_import_no_cycle() -> None:
    script = (
        "from pmarlo.ml.deeptica.trainer import CurriculumConfig\n"
        "from pmarlo.features.deeptica.core.trainer_api import train_deeptica_pipeline\n"
        "assert CurriculumConfig is not None\n"
        "assert train_deeptica_pipeline is not None\n"
    )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
