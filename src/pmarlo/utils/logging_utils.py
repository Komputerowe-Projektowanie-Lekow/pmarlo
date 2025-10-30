from __future__ import annotations

"""Utilities for consistent console-and-log banners and timing helpers."""

import logging
from datetime import timedelta
from dataclasses import dataclass, field
from time import perf_counter
from types import TracebackType
from typing import Literal, Optional, Sequence

BORDER = "=" * 80


def format_duration(seconds: float) -> str:
    """Render a duration in seconds into a human-readable ASCII string."""

    duration = timedelta(seconds=max(seconds, 0.0))
    total_seconds = duration.total_seconds()

    if total_seconds < 1.0:
        return f"{total_seconds * 1000.0:.0f} ms"
    if total_seconds < 60.0:
        return f"{total_seconds:.2f} s"

    days = duration.days
    remaining_seconds = duration.seconds
    hours, remaining_seconds = divmod(remaining_seconds, 3600)
    minutes, seconds_whole = divmod(remaining_seconds, 60)
    seconds_fraction = seconds_whole + duration.microseconds / 1_000_000

    if days == 0 and hours == 0:
        return f"{minutes} min {seconds_fraction:.1f} s"
    if days == 0:
        return f"{hours} h {minutes} min {seconds_fraction:.1f} s"
    return f"{days} d {hours} h {minutes} min"


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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> Literal[False]:
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
