from __future__ import annotations

import os
from typing import Dict, Tuple

from openmm import Platform


def select_platform_and_properties(
    logger, prefer_deterministic: bool = False
) -> Tuple[Platform, Dict[str, str]]:
    platform_properties: Dict[str, str] = {}
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
            }
    try:
        supported = set(platform.getPropertyNames())
        platform_properties = {
            k: v for k, v in platform_properties.items() if k in supported
        }
    except Exception:
        pass
    return platform, platform_properties
