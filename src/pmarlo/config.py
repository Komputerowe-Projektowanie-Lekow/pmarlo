from contextlib import contextmanager
from contextvars import ContextVar

# Feature toggles using ContextVar for per-context overrides
REORDER_STATES: ContextVar[bool] = ContextVar("REORDER_STATES", default=True)
USE_BEAM_SEARCH: ContextVar[bool] = ContextVar("USE_BEAM_SEARCH", default=False)
ALLOW_GAP_REPAIR: ContextVar[bool] = ContextVar("ALLOW_GAP_REPAIR", default=True)
FES_SMOOTHING: ContextVar[bool] = ContextVar("FES_SMOOTHING", default=True)


@contextmanager
def override(var: ContextVar, value):
    """Temporarily override a ContextVar within a scope."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)
