from __future__ import annotations

import os
from typing import Dict, Tuple

from openmm import Platform


def select_platform_and_properties(logger) -> Tuple[Platform, Dict[str, str]]:
    platform_properties: Dict[str, str] = {}
    try:
        platform = Platform.getPlatformByName("CUDA")
        platform_properties = {
            "Precision": "mixed",
            "UseFastMath": "true",
            "DeterministicForces": "false",
        }
        logger.info("Using CUDA (mixed precision, fast math)")
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
            logger.info("Using CPU with all cores")
            platform_properties = {
                "Threads": str(os.cpu_count() or 1),
                "CpuThreads": str(os.cpu_count() or 1),
            }
    try:
        supported = set(platform.getPropertyNames())
        platform_properties = {
            k: v for k, v in platform_properties.items() if k in supported
        }
    except Exception:
        pass
    return platform, platform_properties
