"""Tests for probing replica trajectory metadata during demultiplexing."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Ensure the src directory is importable without loading the full package init.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

pmarlo_stub = types.ModuleType("pmarlo")
pmarlo_stub.__path__ = [str(SRC_ROOT / "pmarlo")]
pmarlo_spec = importlib.util.spec_from_loader("pmarlo", loader=None)
if pmarlo_spec is not None:
    pmarlo_spec.submodule_search_locations = [str(SRC_ROOT / "pmarlo")]
    pmarlo_stub.__spec__ = pmarlo_spec
sys.modules.setdefault("pmarlo", pmarlo_stub)

pmarlo_demux_stub = types.ModuleType("pmarlo.demultiplexing")
pmarlo_demux_stub.__path__ = [str(SRC_ROOT / "pmarlo" / "demultiplexing")]
sys.modules.setdefault("pmarlo.demultiplexing", pmarlo_demux_stub)

pmarlo_transform_stub = types.ModuleType("pmarlo.transform")
pmarlo_transform_stub.__path__ = []
sys.modules.setdefault("pmarlo.transform", pmarlo_transform_stub)

progress_stub = types.ModuleType("pmarlo.transform.progress")
progress_stub.ProgressCB = object  # type: ignore[assignment]


class _ProgressReporter:
    def __init__(self, cb, min_interval_s: float = 0.4):  # pragma: no cover - stub
        self._cb = cb

    def emit(
        self, event: str, info: dict | None = None
    ) -> None:  # pragma: no cover - stub
        if self._cb is not None:
            self._cb(event, info or {})


progress_stub.ProgressReporter = _ProgressReporter
sys.modules.setdefault("pmarlo.transform.progress", progress_stub)

replica_exchange_stub = types.ModuleType("pmarlo.replica_exchange")
replica_exchange_stub.__path__ = [str(SRC_ROOT / "pmarlo" / "replica_exchange")]
replica_exchange_spec = importlib.util.spec_from_loader(
    "pmarlo.replica_exchange", loader=None
)
if replica_exchange_spec is not None:
    replica_exchange_spec.submodule_search_locations = [
        str(SRC_ROOT / "pmarlo" / "replica_exchange")
    ]
    replica_exchange_stub.__spec__ = replica_exchange_spec
sys.modules.setdefault("pmarlo.replica_exchange", replica_exchange_stub)

config_spec = importlib.util.spec_from_file_location(
    "pmarlo.replica_exchange.config",
    SRC_ROOT / "pmarlo" / "replica_exchange" / "config.py",
)
assert config_spec and config_spec.loader is not None
config_module = importlib.util.module_from_spec(config_spec)
sys.modules[config_spec.name] = config_module
config_spec.loader.exec_module(config_module)
replica_exchange_stub.config = config_module

openmm_stub = types.ModuleType("openmm")
openmm_stub.unit = types.SimpleNamespace(picoseconds="picoseconds")
openmm_stub.OpenMMException = type("OpenMMException", (Exception,), {})


class _PlatformStub:
    @staticmethod
    def getPlatformByName(name: str):  # pragma: no cover - stub
        raise openmm_stub.OpenMMException(name)


openmm_stub.Platform = _PlatformStub
sys.modules.setdefault("openmm", openmm_stub)

mdtraj_stub = types.ModuleType("mdtraj")
mdtraj_stub.load_topology = (
    lambda path: types.SimpleNamespace()
)  # pragma: no cover - stub
sys.modules.setdefault("mdtraj", mdtraj_stub)

psutil_stub = types.ModuleType("psutil")
sys.modules.setdefault("psutil", psutil_stub)

openmm_app_stub = types.ModuleType("openmm.app")
openmm_app_stub.PME = object  # type: ignore[assignment]
openmm_app_stub.HBonds = object  # type: ignore[assignment]
openmm_app_stub.ForceField = type("ForceField", (), {})  # type: ignore[assignment]
openmm_app_stub.PDBFile = type("PDBFile", (), {})  # type: ignore[assignment]
openmm_app_stub.Simulation = object  # type: ignore[assignment]
openmm_app_stub.DCDReporter = type("DCDReporter", (), {})  # type: ignore[assignment]
sys.modules.setdefault("openmm.app", openmm_app_stub)

spec = importlib.util.spec_from_file_location(
    "pmarlo.demultiplexing.demux", SRC_ROOT / "pmarlo" / "demultiplexing" / "demux.py"
)
assert spec and spec.loader is not None
demux_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demux_module
spec.loader.exec_module(demux_module)

from pmarlo.io.trajectory_reader import TrajectoryIOError

_probe_replica_info = demux_module._probe_replica_info


class _MissingTrajectoryError(TrajectoryIOError):
    """Trajectory reader error that wraps a missing-file cause."""


class _DummyRemd:
    def __init__(self, paths: list[Path]) -> None:
        self.trajectory_files = paths


def test_probe_marks_missing_file_as_reader_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_replica.dcd"

    class _Reader:
        def probe_length(self, path: str) -> int:  # pragma: no cover - interface stub
            raise FileNotFoundError(path)

    paths, frames, had_error = _probe_replica_info(_DummyRemd([missing]), _Reader())

    assert paths == [str(missing)]
    assert frames == [0]
    assert had_error is True


def test_probe_marks_missing_file_wrapped_in_trajectory_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing_replica_wrapped.dcd"

    class _Reader:
        def probe_length(self, path: str) -> int:  # pragma: no cover - interface stub
            raise _MissingTrajectoryError(path) from FileNotFoundError(path)

    paths, frames, had_error = _probe_replica_info(_DummyRemd([missing]), _Reader())

    assert paths == [str(missing)]
    assert frames == [0]
    assert had_error is True
