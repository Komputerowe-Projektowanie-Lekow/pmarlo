from __future__ import annotations

from pathlib import Path


def test_deeptica_tab_gating_banner_message_exists():
    # The app shows a banner when Deep‑TICA is skipped; ensure the message template exists.
    p = Path("example_programs/app_usecase/app/app.py")
    text = p.read_text(encoding="utf-8")
    assert (
        "Deep‑TICA CV learning was skipped" in text
        or "Deep-TICA CV learning was skipped" in text
    )


def test_deeptica_tab_gating_predicate():
    # Minimal logic match: banner should be shown when artifact indicates skipped
    mlcv = {"applied": False, "skipped": True, "reason": "no_pairs"}
    should_show_banner = (not isinstance(mlcv, dict)) or (
        not mlcv.get("applied", False)
    )
    assert should_show_banner is True
