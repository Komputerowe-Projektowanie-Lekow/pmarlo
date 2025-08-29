import os
from dataclasses import dataclass


@dataclass
class ConfigVar:
    """Simple configuration helper using environment variables."""

    env: str
    default: bool = False

    def get(self) -> bool:
        val = os.getenv(self.env)
        if val is None:
            return self.default
        return val.lower() not in {"0", "false", "no"}


FES_SMOOTHING = ConfigVar("PMARLO_FES_SMOOTHING", False)
REORDER_STATES = ConfigVar("PMARLO_REORDER_STATES", False)
