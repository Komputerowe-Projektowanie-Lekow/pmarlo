"""Canonical OpenMM platform selection for replica-exchange simulations."""

from __future__ import annotations

import os
from typing import Dict, Tuple

from openmm import Platform


def select_platform_and_properties(
    logger, prefer_deterministic: bool = False
) -> Tuple[Platform, Dict[str, str]]:
    """Return a deterministic OpenMM platform selection.

    The resolution order is:

    1. ``OPENMM_PLATFORM`` / ``PMARLO_FORCE_PLATFORM`` environment variables.
    2. ``Reference`` when ``prefer_deterministic`` is ``True``.
    3. ``CUDA`` otherwise.

    Missing platforms raise immediately instead of silently falling back.
    """

    forced = os.getenv("OPENMM_PLATFORM") or os.getenv("PMARLO_FORCE_PLATFORM")
    if forced:
        platform_name = forced
        logger.info("Using forced OpenMM platform %s", forced)
    elif prefer_deterministic:
        platform_name = "Reference"
        logger.info("Using Reference platform for deterministic execution")
    else:
        platform_name = "CUDA"
        logger.info("Using CUDA platform")

    platform = Platform.getPlatformByName(platform_name)

    properties: Dict[str, str] = {}
    if platform_name == "CUDA":
        properties = {
            "Precision": "single" if prefer_deterministic else "mixed",
            "UseFastMath": "false" if prefer_deterministic else "true",
            "DeterministicForces": "true" if prefer_deterministic else "false",
            "DeviceIndex": os.getenv("PMARLO_CUDA_DEVICE", "0"),
        }
    elif platform_name == "CPU":
        threads = os.getenv("PMARLO_CPU_THREADS")
        properties = {
            "Threads": threads or "",
            "CpuThreads": threads or "",
        }
        if prefer_deterministic:
            properties["DeterministicForces"] = "true"

    supported = set(platform.getPropertyNames())
    filtered = {k: v for k, v in properties.items() if k in supported and v}
    return platform, filtered
