from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from pmarlo.io import trajectory


def test_dcd_plugin_quiet_by_default():
    """Ensure noisy dcdplugin messages do not reach stdout."""

    traj = Path("tests/data/traj.dcd")
    pdb = Path("tests/data/3gd8-fixed.pdb")

    buf = io.StringIO()
    with redirect_stdout(buf):
        # Only need to trigger a single chunk to reproduce the banner
        for _ in trajectory.iterload(str(traj), top=str(pdb), chunk=5):
            break
    out = buf.getvalue().splitlines()
    assert not any(line.startswith("dcdplugin)") for line in out)
