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
    2. Auto-select fastest available platform (CUDA > CPU > Reference)
    3. Enable deterministic flags if ``prefer_deterministic`` is ``True``.

    Note: Reference platform is ONLY used when explicitly forced via environment
    variable or when no other platform is available. It is 10-100x slower than
    CPU and should never be used for production simulations.
    """

    forced = os.getenv("OPENMM_PLATFORM") or os.getenv("PMARLO_FORCE_PLATFORM")
    if forced:
        platform_name = forced
        logger.info("Using forced OpenMM platform: %s", forced)
    else:
        # Auto-select fastest available platform
        available_platforms = [Platform.getPlatform(i).getName() 
                               for i in range(Platform.getNumPlatforms())]
        
        # Prefer CUDA > CPU > Reference (Reference is VERY slow, avoid if possible)
        if "CUDA" in available_platforms:
            platform_name = "CUDA"
            logger.info("Using CUDA platform%s", " (deterministic mode)" if prefer_deterministic else "")
        elif "CPU" in available_platforms:
            platform_name = "CPU"
            logger.info("Using CPU platform%s", " (deterministic mode)" if prefer_deterministic else "")
        elif "Reference" in available_platforms:
            platform_name = "Reference"
            logger.warning("Using Reference platform - this is VERY slow (10-100x slower than CPU). "
                          "Install OpenMM with CPU support for better performance.")
        else:
            # This should never happen but handle gracefully
            platform_name = "CUDA"
            logger.warning("No standard OpenMM platform found, attempting CUDA as fallback")

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
        threads = os.getenv("PMARLO_CPU_THREADS", "0")  # 0 = auto-detect
        properties = {}
        if threads and threads != "0":
            properties["Threads"] = threads
        if prefer_deterministic:
            properties["DeterministicForces"] = "true"
    elif platform_name == "Reference":
        # Reference has no configurable properties
        properties = {}

    supported = set(platform.getPropertyNames())
    filtered = {k: v for k, v in properties.items() if k in supported and v}
    return platform, filtered
