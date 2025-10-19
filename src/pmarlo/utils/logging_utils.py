from __future__ import annotations

"""Utilities for consistent console-and-log banners and timing helpers."""

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional, Sequence

BORDER = "=" * 80


def format_duration(seconds: float) -> str:
    """Render a duration in seconds into a human-readable ASCII string."""
    if seconds < 0.0:
        seconds = 0.0
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 1.0:
        return f"{seconds:.2f} s"
    hours, minutes_int = divmod(int(minutes), 60)
    if hours == 0:
        return f"{minutes_int} min {remainder:.1f} s"
    days, hours = divmod(hours, 24)
    if days == 0:
        return f"{hours} h {minutes_int} min {remainder:.1f} s"
    return f"{days} d {hours} h {minutes_int} min"


def format_stage_header(
    stage_label: str,
    *,
    index: int | None = None,
    total: int | None = None,
) -> str:
    """Return a normalized header for a stage banner."""
    normalized = stage_label.strip().upper()
    if index is not None and total is not None:
        return f"STAGE {index}/{total}: {normalized}"
    return normalized


def emit_banner(
    message: str,
    *,
    logger: logging.Logger,
    details: Sequence[str] | None = None,
    newline_before: bool = True,
) -> None:
    """Print and log a banner with optional detail lines."""
    prefix = "\n" if newline_before else ""
    print(prefix + BORDER, flush=True)
    print(message, flush=True)
    print(BORDER, flush=True)
    if details:
        for line in details:
            print(line, flush=True)
    print(BORDER + "\n", flush=True)

    logger.info(BORDER)
    logger.info(message)
    logger.info(BORDER)
    if details:
        for line in details:
            logger.info(line)
    logger.info(BORDER)


def announce_stage_start(
    stage_label: str,
    *,
    logger: logging.Logger,
    index: int | None = None,
    total: int | None = None,
    details: Sequence[str] | None = None,
) -> None:
    """Emit a standard banner for the beginning of a stage."""
    header = format_stage_header(stage_label, index=index, total=total)
    emit_banner(header, logger=logger, details=details)


def announce_stage_complete(
    stage_label: str,
    *,
    logger: logging.Logger,
    details: Sequence[str] | None = None,
) -> None:
    """Emit a standard banner for stage completion."""
    emit_banner(
        f"{stage_label.strip().upper()} COMPLETE", logger=logger, details=details
    )


def announce_stage_failed(
    stage_label: str,
    *,
    logger: logging.Logger,
    details: Sequence[str] | None = None,
) -> None:
    """Emit a standard banner for stage failure."""
    emit_banner(f"{stage_label.strip().upper()} FAILED", logger=logger, details=details)


def announce_stage_cancelled(
    stage_label: str,
    *,
    logger: logging.Logger,
    details: Sequence[str] | None = None,
) -> None:
    """Emit a standard banner for stage cancellation."""
    emit_banner(
        f"{stage_label.strip().upper()} CANCELLED", logger=logger, details=details
    )


@dataclass
class StageTimer:
    """Context manager that measures execution time and logs completion."""

    label: str
    logger: logging.Logger
    print_on_complete: bool = True
    start_message: Optional[str] = None
    details: Sequence[str] | None = None

    _start: float = field(init=False, default=0.0)
    elapsed: float = field(init=False, default=0.0)

    def __enter__(self) -> "StageTimer":
        self._start = perf_counter()
        if self.start_message:
            print(self.start_message, flush=True)
            self.logger.info(self.start_message)
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        self.elapsed = perf_counter() - self._start
        status = "completed" if exc is None else "failed"
        message = f"{self.label} {status} in {format_duration(self.elapsed)}."
        if self.print_on_complete:
            print(message, flush=True)
        log_level = logging.ERROR if exc is not None else logging.INFO
        self.logger.log(log_level, message)
        if exc is None and self.details:
            for line in self.details:
                self.logger.info(line)
        return False
