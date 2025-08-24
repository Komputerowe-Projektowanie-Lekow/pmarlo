"""
Lightweight in-terminal progress bar with ETA.

Designed to be dependency-free and safe for long-running loops. Prints using
carriage returns and flushes on each update. Provides a helper to terminate the
current progress line before emitting regular log lines.
"""

from __future__ import annotations

import sys
import time
from typing import IO, Optional


class ProgressPrinter:
    def __init__(
        self, total: int, bar_width: int = 30, stream: IO[str] | None = None
    ) -> None:
        self.total = max(1, int(total))
        self.bar_width = max(10, int(bar_width))
        self.start_t = time.monotonic()
        self.last_percent_drawn = -1
        self.stream: IO[str] = stream or sys.stdout
        self._active = False

    def _format_eta(self, remaining_seconds: float) -> str:
        remaining_seconds = max(0, int(remaining_seconds))
        hours, rem = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

    def draw(self, current: int, suffix: Optional[str] = None) -> None:
        current = max(0, min(self.total, int(current)))
        frac = current / self.total
        percent = int(frac * 100)
        if percent == self.last_percent_drawn:
            return
        self.last_percent_drawn = percent
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        elapsed = time.monotonic() - self.start_t
        eta = (
            (elapsed / max(1e-9, current)) * (self.total - current)
            if current > 0
            else 0.0
        )
        eta_str = self._format_eta(eta)
        suffix_str = f" {suffix}" if suffix else ""
        print(
            f"\r[{bar}] {percent}%/100% ETA {eta_str}{suffix_str}",
            end="",
            flush=True,
            file=self.stream,
        )
        self._active = True

    def newline_if_active(self) -> None:
        if self._active:
            # End the current progress line so following logs appear on a new line
            print("", file=self.stream)
            self._active = False

    def close(self) -> None:
        self.draw(self.total)
        print("", flush=True, file=self.stream)
        self._active = False
