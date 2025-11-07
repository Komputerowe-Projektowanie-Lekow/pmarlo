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
    2. Auto-select fastest available platform (CUDA > CPU)
    3. Enable deterministic flags if ``prefer_deterministic`` is ``True``.

    Raises RuntimeError if neither CUDA nor CPU platforms are available.
    """

    forced = os.getenv("OPENMM_PLATFORM") or os.getenv("PMARLO_FORCE_PLATFORM")
    if forced:
        platform_name = forced
        logger.info("Using forced OpenMM platform: %s", forced)
    else:
        # Auto-select fastest available platform
        available_platforms = [
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]

        if not available_platforms:
            raise RuntimeError(
                "No OpenMM platforms are available. Install at least one backend (e.g. CPU or CUDA)."
            )

        # Prefer CUDA > CPU (no fallback to Reference - it's too slow)
        if "CUDA" in available_platforms:
            platform_name = "CUDA"
            logger.info(
                "Using CUDA platform%s",
                " (deterministic mode)" if prefer_deterministic else "",
            )
        elif "CPU" in available_platforms:
            platform_name = "CPU"
            logger.info(
                "Using CPU platform%s",
                " (deterministic mode)" if prefer_deterministic else "",
            )
        else:
            raise RuntimeError(
                f"No suitable OpenMM platform available. Found: {available_platforms}. "
                "Please install OpenMM with CUDA or CPU support. "
                "The Reference platform is too slow for production use."
            )

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
        # CPU can be deterministic AND fast
        # Note: DeterministicForces flag is NOT needed for determinism on CPU
        # (integrator random seed is sufficient) and it disables multicore
        # optimizations. We only set Threads to allow user control.
        threads = os.getenv("PMARLO_CPU_THREADS", "0")  # 0 = auto-detect
        properties = {}
        if threads and threads != "0":
            properties["Threads"] = threads
    elif platform_name == "Reference":
        # Reference has no configurable properties
        properties = {}

    supported = set(platform.getPropertyNames())
    filtered = {k: v for k, v in properties.items() if k in supported and v}
    return platform, filtered
