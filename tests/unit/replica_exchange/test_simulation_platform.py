# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for explicit OpenMM platform selection in :mod:`pmarlo.replica_exchange`."""

from pathlib import Path

import pytest

from pmarlo.replica_exchange import _simulation_full


ASSET_PDB = Path("tests/_assets/3gd8-fixed.pdb")


@pytest.fixture()
def simulation(tmp_path):
    """Create a simulation object pointing to the real 3gd8-fixed PDB asset."""

    def _factory(platform: str):
        return _simulation_full.Simulation(
            pdb_file=str(ASSET_PDB),
            output_dir=str(tmp_path),
            platform=platform,
        )

    return _factory


def test_setup_platform_rejects_unsupported_backend(simulation):
    sim = simulation("OpenCL")

    with pytest.raises(ValueError, match="Unsupported OpenMM platform"):
        sim._setup_platform()


def test_setup_platform_raises_when_backend_unavailable(simulation, monkeypatch):
    sim = simulation("CUDA")

    class DummyPlatformModule:
        @staticmethod
        def getPlatformByName(name):
            raise Exception("unavailable")

    monkeypatch.setattr(_simulation_full.openmm, "Platform", DummyPlatformModule)

    with pytest.raises(RuntimeError, match="unavailable"):
        sim._setup_platform()


def test_setup_platform_allows_cpu(simulation, monkeypatch):
    sim = simulation("CPU")

    class DummyPlatform:
        def __init__(self, name):
            self.name = name
            self.properties: dict[str, str] = {}

        def setPropertyDefaultValue(self, key, value):  # pragma: no cover - CPU path
            self.properties[key] = value

    class DummyPlatformModule:
        @staticmethod
        def getPlatformByName(name):
            assert name == "CPU"
            return DummyPlatform(name)

    monkeypatch.setattr(_simulation_full.openmm, "Platform", DummyPlatformModule)

    sim._setup_platform()

    assert isinstance(sim.platform, DummyPlatform)
    assert sim.platform.name == "CPU"


def test_setup_platform_sets_cuda_precision(simulation, monkeypatch):
    sim = simulation("CUDA")

    class DummyPlatform:
        def __init__(self):
            self.properties: dict[str, str] = {}

        def setPropertyDefaultValue(self, key, value):
            self.properties[key] = value

    class DummyPlatformModule:
        @staticmethod
        def getPlatformByName(name):
            assert name == "CUDA"
            return DummyPlatform()

    monkeypatch.setattr(_simulation_full.openmm, "Platform", DummyPlatformModule)

    sim._setup_platform()

    assert sim.platform.properties == {"Precision": "mixed"}
