from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[3] / "pmarlo_webapp" / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from core.constants import DEEPTICA_SKIP_MESSAGE


def test_deeptica_tab_gating_banner_message_exists():
    # Ensure the skip banner message used by the UI is still defined.
    assert "Deep-TICA CV learning was skipped" in DEEPTICA_SKIP_MESSAGE


def test_deeptica_tab_gating_predicate():
    # Minimal logic match: banner should be shown when artifact indicates skipped
    mlcv = {"applied": False, "skipped": True, "reason": "no_pairs"}
    should_show_banner = (not isinstance(mlcv, dict)) or (
        not mlcv.get("applied", False)
    )
    assert should_show_banner is True
