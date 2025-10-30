from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


@dataclass
class _LogCollector:
    messages: list[tuple[str, tuple]]

    def info(self, msg: str, *args) -> None:  # pragma: no cover - simple proxy
        self.messages.append((msg, args))

    def warning(self, msg: str, *args) -> None:  # pragma: no cover - simple proxy
        self.messages.append((msg, args))


class _DummyPlatform:
    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name

    def getPropertyNames(self):  # pragma: no cover - minimal API
        return []


class _DummyPlatformModule:
    def __init__(self, names: list[str]):
        self._names = names

    def getPlatform(self, index: int) -> _DummyPlatform:
        return _DummyPlatform(self._names[index])

    def getNumPlatforms(self) -> int:
        return len(self._names)

    def getPlatformByName(self, name: str) -> _DummyPlatform:
        return _DummyPlatform(name)


class _ImportStubPlatformModule:
    def getPlatform(self, index: int) -> _DummyPlatform:  # pragma: no cover - unused
        raise RuntimeError("no platforms stubbed")

    def getNumPlatforms(self) -> int:
        return 0

    def getPlatformByName(self, name: str) -> _DummyPlatform:  # pragma: no cover - unused
        raise RuntimeError("no platforms stubbed")


@pytest.fixture()
def platform_selector_module(monkeypatch):
    module_name = "platform_selector_under_test"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pmarlo"
        / "replica_exchange"
        / "platform_selector.py"
    )

    stub_openmm = ModuleType("openmm")
    stub_openmm.Platform = _ImportStubPlatformModule()
    monkeypatch.setitem(sys.modules, "openmm", stub_openmm)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selects_first_available_when_preferred_missing(platform_selector_module, monkeypatch):
    dummy_module = _DummyPlatformModule(["OpenCL"])
    monkeypatch.setattr(platform_selector_module, "Platform", dummy_module)

    log = _LogCollector(messages=[])
    platform, properties = platform_selector_module.select_platform_and_properties(log)

    assert isinstance(platform, _DummyPlatform)
    assert platform.getName() == "OpenCL"
    assert properties == {}


def test_raises_when_no_platforms_present(platform_selector_module, monkeypatch):
    dummy_module = _DummyPlatformModule([])
    monkeypatch.setattr(platform_selector_module, "Platform", dummy_module)

    log = _LogCollector(messages=[])

    with pytest.raises(RuntimeError, match="No OpenMM platforms are available"):
        platform_selector_module.select_platform_and_properties(log)
