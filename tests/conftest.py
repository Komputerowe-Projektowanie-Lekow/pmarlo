# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pytest configuration and fixtures for PMARLO tests."""

import shutil
import sys
import tempfile
import types
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

import pytest

# Only install torch stub if real torch is not available
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _TORCH_STUB = types.ModuleType("torch")
        _TORCH_STUB.manual_seed = lambda *args, **kwargs: None
        _TORCH_STUB.use_deterministic_algorithms = lambda *args, **kwargs: None
        sys.modules["torch"] = _TORCH_STUB

_OPENMM_MODULE = sys.modules.get("openmm")
if _OPENMM_MODULE is None:
    _OPENMM_MODULE = types.ModuleType("openmm")
    _OPENMM_MODULE.__pmarlo_stub__ = True  # type: ignore[attr-defined]
    sys.modules["openmm"] = _OPENMM_MODULE
else:
    setattr(
        _OPENMM_MODULE,
        "__pmarlo_stub__",
        getattr(_OPENMM_MODULE, "__pmarlo_stub__", False),
    )
if getattr(_OPENMM_MODULE, "__spec__", None) is None:
    _OPENMM_MODULE.__spec__ = ModuleSpec("openmm", loader=None)
_OPENMM_MODULE.unit = getattr(
    _OPENMM_MODULE, "unit", types.SimpleNamespace(picoseconds="picoseconds")
)
if not hasattr(_OPENMM_MODULE, "OpenMMException"):
    class _OpenMMException(Exception):  # pragma: no cover - stub
        pass

    _OPENMM_MODULE.OpenMMException = _OpenMMException
if not hasattr(_OPENMM_MODULE, "Integrator"):
    _OPENMM_MODULE.Integrator = type("Integrator", (), {})  # type: ignore[assignment]
if not hasattr(_OPENMM_MODULE, "System"):
    _OPENMM_MODULE.System = type("System", (), {})  # type: ignore[assignment]
if not hasattr(_OPENMM_MODULE, "XmlSerializer"):
    _OPENMM_MODULE.XmlSerializer = type(
        "XmlSerializer", (), {"load": staticmethod(lambda *_args, **_kwargs: None)}
    )  # type: ignore[assignment]
if not hasattr(_OPENMM_MODULE, "Platform"):
    class _PlatformStub:
        @staticmethod
        def getPlatformByName(name: str) -> None:  # pragma: no cover - stub
            raise _OPENMM_MODULE.OpenMMException(name)

    _OPENMM_MODULE.Platform = _PlatformStub

_OPENMM_APP_MODULE = sys.modules.get("openmm.app")
if _OPENMM_APP_MODULE is None:
    _OPENMM_APP_MODULE = types.ModuleType("openmm.app")
    sys.modules["openmm.app"] = _OPENMM_APP_MODULE

_OPENMM_APP_DEFAULTS = {
    "PME": object,
    "HBonds": object,
    "Modeller": type("Modeller", (), {}),
    "ForceField": type("ForceField", (), {}),
    "PDBFile": type("PDBFile", (), {}),
    "Simulation": type("Simulation", (), {}),
    "DCDReporter": type("DCDReporter", (), {}),
}
for _name, _value in _OPENMM_APP_DEFAULTS.items():
    if not hasattr(_OPENMM_APP_MODULE, _name):
        setattr(_OPENMM_APP_MODULE, _name, _value)
_OPENMM_METAD_MODULE = sys.modules.get("openmm.app.metadynamics")
if _OPENMM_METAD_MODULE is None:
    _OPENMM_METAD_MODULE = types.ModuleType("openmm.app.metadynamics")
    _OPENMM_METAD_MODULE.BiasVariable = type("BiasVariable", (), {})  # type: ignore[assignment]
    _OPENMM_METAD_MODULE.Metadynamics = type("Metadynamics", (), {})  # type: ignore[assignment]
    sys.modules["openmm.app.metadynamics"] = _OPENMM_METAD_MODULE

_OPENMM_UNIT_MODULE = sys.modules.get("openmm.unit")
if _OPENMM_UNIT_MODULE is None:
    _OPENMM_UNIT_MODULE = types.ModuleType("openmm.unit")
    sys.modules["openmm.unit"] = _OPENMM_UNIT_MODULE
if not hasattr(_OPENMM_UNIT_MODULE, "picoseconds"):
    _OPENMM_UNIT_MODULE.picoseconds = "picoseconds"  # type: ignore[assignment]
if not hasattr(_OPENMM_UNIT_MODULE, "kilojoules_per_mole"):
    _OPENMM_UNIT_MODULE.kilojoules_per_mole = 1.0  # type: ignore[assignment]
if not hasattr(_OPENMM_UNIT_MODULE, "kelvin"):
    _OPENMM_UNIT_MODULE.kelvin = 1.0  # type: ignore[assignment]
if not hasattr(_OPENMM_UNIT_MODULE, "MOLAR_GAS_CONSTANT_R"):
    _OPENMM_UNIT_MODULE.MOLAR_GAS_CONSTANT_R = 8.31446261815324  # type: ignore[assignment]
if not hasattr(_OPENMM_UNIT_MODULE, "dimensionless"):
    _OPENMM_UNIT_MODULE.dimensionless = 1.0  # type: ignore[assignment]
_OPENMM_MODULE.unit = _OPENMM_UNIT_MODULE
_OPENMM_QUANTITY_MODULE = sys.modules.get("openmm.unit.quantity")
if _OPENMM_QUANTITY_MODULE is None:
    _OPENMM_QUANTITY_MODULE = types.ModuleType("openmm.unit.quantity")
    _OPENMM_QUANTITY_MODULE.Quantity = type("Quantity", (), {})  # type: ignore[assignment]
    sys.modules["openmm.unit.quantity"] = _OPENMM_QUANTITY_MODULE
if not hasattr(_OPENMM_UNIT_MODULE, "Quantity"):
    _OPENMM_UNIT_MODULE.Quantity = _OPENMM_QUANTITY_MODULE.Quantity  # type: ignore[assignment]
if not hasattr(_OPENMM_UNIT_MODULE, "quantity"):
    _OPENMM_UNIT_MODULE.quantity = types.SimpleNamespace(
        Quantity=_OPENMM_QUANTITY_MODULE.Quantity
    )

TESTS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = TESTS_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Folder -> default markers that should apply to every test collected under it.
FOLDER_MARKERS = {
    "tests/unit/app/": ["unit", "workflow"],
    "tests/unit/api/": ["unit", "workflow"],
    "tests/unit/cv/": ["unit", "cv"],
    "tests/unit/data/": ["unit", "data"],
    "tests/unit/demultiplexing/": ["unit", "demux"],
    "tests/unit/experiments/": ["unit", "experiments"],
    "tests/unit/features/": ["unit", "features"],
    "tests/unit/io/": ["unit", "io"],
    "tests/unit/markov_state_model/": ["unit", "msm"],
    "tests/unit/protein/": ["unit", "protein"],
    "tests/unit/reduce/": ["unit", "reduce"],
    "tests/unit/replica_exchange/": ["unit", "replica"],
    "tests/unit/reporting/": ["unit", "reporting"],
    "tests/unit/results/": ["unit", "results"],
    "tests/unit/transform/": ["unit", "transform"],
    "tests/unit/utils/": ["unit", "utils"],
    "tests/unit/workflow/": ["unit", "workflow"],
    "tests/integration/replica_exchange/": ["integration", "replica"],
    "tests/integration/workflow/": ["integration", "workflow"],
    "tests/integration/smoke/": ["integration"],
    "tests/perf/": ["perf", "slow"],
}


def _normalize_path(path: Path) -> str:
    """Return a forward-slash path for prefix matching."""
    return str(path).replace("\\", "/")


def _apply_folder_markers(item: pytest.Item) -> None:
    """Attach default markers based on the test file location."""
    normalized = _normalize_path(Path(str(item.fspath)))
    applied: set[str] = set()
    for folder, markers in FOLDER_MARKERS.items():
        if folder in normalized:
            for marker in markers:
                if marker not in applied:
                    item.add_marker(getattr(pytest.mark, marker))
                    applied.add(marker)


def _parse_focus_option(raw: str) -> set[str]:
    return {chunk.strip() for chunk in raw.split(",") if chunk.strip()}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--focus",
        action="store",
        default="",
        help="Comma-separated domain markers (e.g. data,io,msm). Only matching tests run.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    focus = _parse_focus_option(config.getoption("--focus"))
    deselected: list[pytest.Item] = []
    selected: list[pytest.Item] = []

    for item in items:
        _apply_folder_markers(item)
        if not focus:
            continue
        tags = {mark.name for mark in item.iter_markers()}
        if focus.intersection(tags) or "all" in focus:
            selected.append(item)
        else:
            deselected.append(item)

    if focus and deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers that are not in pyproject."""
    config.addinivalue_line("markers", "pdbfixer: mark test as requiring PDBFixer")


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "_assets"


@pytest.fixture
def test_pdb_file(test_data_dir: Path) -> Path:
    """Path to test PDB file."""
    return test_data_dir / "3gd8.pdb"


@pytest.fixture
def test_fixed_pdb_file(test_data_dir: Path) -> Path:
    """Path to test fixed PDB file."""
    return test_data_dir / "3gd8-fixed.pdb"


@pytest.fixture
def test_trajectory_file(test_data_dir: Path) -> Path:
    """Path to test trajectory file."""
    return test_data_dir / "traj.dcd"


@pytest.fixture
def temp_output_dir() -> Iterable[Path]:
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def skip_if_no_openmm() -> bool:
    """Skip tests if OpenMM is not available."""
    return find_spec("openmm") is None


@pytest.fixture
def damaged_pdb_file(tmp_path: Path) -> Path:
    """Create a deliberately damaged PDB file."""
    path = tmp_path / "damaged.pdb"
    path.write_text("ATOM      1  N   ALA A   1\nEND\n")
    return path


@pytest.fixture
def nan_pdb_file(tmp_path: Path) -> Path:
    """Create a PDB file containing NaN coordinates."""
    from openmm import Vec3, unit  # type: ignore import
    from openmm.app import PDBFile, Topology, element  # type: ignore import

    path = tmp_path / "nan.pdb"

    top = Topology()
    chain = top.addChain("A")
    res = top.addResidue("ALA", chain)
    top.addAtom("N", element.get_by_symbol("N"), res)
    top.addAtom("CA", element.get_by_symbol("C"), res)
    positions = unit.Quantity(
        [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)],
        unit.nanometer,
    )
    with open(path, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(top, positions, handle)

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "N":
            line = line[:30] + f"{'NaN':>8}" + line[38:]
        elif line.startswith("ATOM") and line[12:16].strip() == "CA":
            line = line[:38] + f"{'NaN':>8}" + line[46:]
        new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return path
