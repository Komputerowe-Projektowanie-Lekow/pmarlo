from __future__ import annotations

import os
from typing import Dict, Tuple

from openmm import Platform


def select_platform_and_properties(  # noqa: C901 - small decision tree by platform
    logger, prefer_deterministic: bool = False
) -> Tuple[Platform, Dict[str, str]]:
    platform_properties: Dict[str, str] = {}
    # Allow environment override to force a specific platform for troubleshooting
    # or CI determinism. Recognizes OPENMM_PLATFORM or PMARLO_FORCE_PLATFORM.
    forced = os.getenv("OPENMM_PLATFORM") or os.getenv("PMARLO_FORCE_PLATFORM")
    if forced:
        try:
            platform = Platform.getPlatformByName(str(forced))
            logger.info(f"Using forced platform: {forced}")
            return platform, platform_properties
        except Exception:
            logger.info(
                f"Requested platform '{forced}' not available; falling back to auto-detect"
            )
    try:
        platform = Platform.getPlatformByName("CUDA")
        platform_properties = {
            "Precision": "single" if prefer_deterministic else "mixed",
            # DeterministicForces and fast math settings influence reproducibility
            "UseFastMath": "false" if prefer_deterministic else "true",
            "DeterministicForces": "true" if prefer_deterministic else "false",
            # Pin to a single device for stability in CI
            "DeviceIndex": "0",
        }
        msg = (
            "Using CUDA (mixed precision, deterministic forces)"
            if prefer_deterministic
            else "Using CUDA (mixed precision, fast math)"
        )
        logger.info(msg)
    except Exception:
        try:
            try:
                platform = Platform.getPlatformByName("HIP")
                logger.info("Using HIP (AMD GPU)")
            except Exception:
                platform = Platform.getPlatformByName("OpenCL")
                logger.info("Using OpenCL")
        except Exception:
            # Prefer Reference platform for determinism when available (slow but stable)
            if prefer_deterministic:
                try:
                    platform = Platform.getPlatformByName("Reference")
                    logger.info("Using Reference platform for deterministic run")
                    return platform, {}
                except Exception:
                    pass
            platform = Platform.getPlatformByName("CPU")
            # Default to a single thread for deterministic tests; allow override via env
            threads = os.getenv("PMARLO_CPU_THREADS") or (
                "1" if prefer_deterministic else "0"
            )
            # Some OpenMM builds use either "Threads" or "CpuThreads"
            logger.info(f"Using CPU with {threads or 'default'} thread(s)")
            platform_properties = {
                "Threads": threads,
                "CpuThreads": threads,
                # Ask for deterministic forces where available
                "DeterministicForces": "true" if prefer_deterministic else "false",
                # Be conservative on math optimizations where supported
                "UseFastMath": "false" if prefer_deterministic else "true",
            }
    try:
        supported = set(platform.getPropertyNames())
        platform_properties = {
            k: v for k, v in platform_properties.items() if k in supported
        }
    except Exception:
        pass
    return platform, platform_properties
