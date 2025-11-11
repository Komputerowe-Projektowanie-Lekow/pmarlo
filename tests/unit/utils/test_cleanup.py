from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo.utils.cleanup import prune_workspace


def test_prune_workspace_requires_standard_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prune_workspace(tmp_path, dry_run=True)
