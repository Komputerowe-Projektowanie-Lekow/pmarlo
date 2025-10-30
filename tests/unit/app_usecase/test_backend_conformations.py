from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

np = pytest.importorskip("numpy")

from pmarlo.conformations.results import Conformation, ConformationSet, TPTResult
from pmarlo_webapp.app.backend import (
    ConformationsConfig,
    WorkflowBackend,
    WorkspaceLayout,
)


@pytest.fixture
def _workspace(tmp_path: Path) -> WorkspaceLayout:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "workspace",
        sims_dir=tmp_path / "workspace" / "sims",
        shards_dir=tmp_path / "workspace" / "shards",
        models_dir=tmp_path / "workspace" / "models",
        bundles_dir=tmp_path / "workspace" / "bundles",
        logs_dir=tmp_path / "workspace" / "logs",
        state_path=tmp_path / "workspace" / "state.json",
    )
    layout.ensure()
    return layout


@pytest.fixture
def _fake_dataset(tmp_path: Path, _workspace: WorkspaceLayout) -> Dict[str, Any]:
    traj_path = _workspace.workspace_dir / "trajectories" / "traj_00.dcd"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"")
    return {
        "X": np.ones((6, 3), dtype=float),
        "__shards__": [
            {
                "trajectories": [str(traj_path)],
            }
        ],
    }


def _patch_common_conformation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    workspace: WorkspaceLayout,
    dataset: Dict[str, Any],
    *,
    patch_reduce: bool = True,
) -> List[str]:
    called: List[str] = []

    def fake_load_shards_as_dataset(_paths: List[Path]) -> Dict[str, Any]:
        called.append("load_dataset")
        return dataset

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.load_shards_as_dataset",
        fake_load_shards_as_dataset,
    )

    if patch_reduce:
        monkeypatch.setattr(
            "pmarlo_webapp.app.backend.reduce_features",
            lambda feats, method, lag, n_components: feats,
        )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.cluster_microstates",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=np.array([0, 1, 0, 1, 0, 1], dtype=int)
        ),
    )

    class _FakeTraj:
        def __init__(self, n_frames: int) -> None:
            self._n = n_frames

        def __len__(self) -> int:  # pragma: no cover - trivial
            return self._n

    def fake_md_load(path: str, top: str) -> _FakeTraj:
        called.append(f"load:{Path(path).name}")
        return _FakeTraj(3)

    def fake_md_join(trajs: List[_FakeTraj]) -> _FakeTraj:
        called.append("join")
        total = sum(len(t) for t in trajs)
        return _FakeTraj(total)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.md.load",
        fake_md_load,
    )
    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.md.join",
        fake_md_join,
    )

    def fake_plot_tpt_summary(_tpt_result: TPTResult, output_dir: str) -> None:
        called.append("plot")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("committors", "flux_network", "pathways"):
            (out / f"{name}.png").write_bytes(b"png")

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.plot_tpt_summary",
        fake_plot_tpt_summary,
    )

    topology_path = workspace.workspace_dir / "app_intputs" / "test.pdb"
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text("END\n", encoding="utf-8")

    return called


@pytest.mark.unit
def test_conformations_requires_topology_path(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    _patch_common_conformation_dependencies(monkeypatch, _workspace, _fake_dataset)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    def fake_find_conformations(**_kwargs: Any) -> ConformationSet:
        raise AssertionError(
            "find_conformations should not be called when topology is missing"
        )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        fake_find_conformations,
    )

    config = ConformationsConfig(topology_pdb=None)
    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is not None
    assert "topology" in result.error.lower()


@pytest.mark.unit
def test_conformations_rejects_non_reversible_transition_matrix(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    calls = _patch_common_conformation_dependencies(
        monkeypatch, _workspace, _fake_dataset
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.9, 0.1], [0.4, 0.6]], dtype=float),
            np.array([0.6, 0.4], dtype=float),
        ),
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        lambda **_kwargs: None,
    )

    config = ConformationsConfig(topology_pdb=Path("app_intputs/test.pdb"))
    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is not None
    assert "reversible" in result.error.lower()
    assert "plot" not in calls  # analysis should abort before plotting


@pytest.mark.unit
def test_conformations_successful_run_uses_conformation_set_api(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    calls = _patch_common_conformation_dependencies(
        monkeypatch, _workspace, _fake_dataset
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.85, 0.15], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    source_states = np.array([0])
    sink_states = np.array([1])
    pathways = [[0, 1]]
    flux = np.array([[0.0, 0.2], [0.0, 0.0]])
    net_flux = np.array([[0.0, 0.2], [-0.2, 0.0]])
    tpt_result = TPTResult(
        source_states=source_states,
        sink_states=sink_states,
        forward_committor=np.array([0.0, 1.0]),
        backward_committor=np.array([1.0, 0.0]),
        flux_matrix=flux,
        net_flux=net_flux,
        total_flux=0.2,
        rate=0.05,
        mfpt=10.0,
        pathways=pathways,
        pathway_fluxes=np.array([0.2]),
        bottleneck_states=np.array([0]),
    )

    conformations = [
        Conformation(
            conformation_type="metastable",
            state_id=0,
            macrostate_id=0,
            frame_index=0,
            population=0.5,
            free_energy=1.0,
            metadata={"microstate_ids": [0, 1]},
            structure_path="metastable_0.pdb",
        ),
        Conformation(
            conformation_type="transition",
            state_id=1,
            frame_index=1,
            population=0.2,
            free_energy=2.0,
            committor=0.5,
            structure_path="transition_1.pdb",
        ),
    ]

    def fake_find_conformations(**_kwargs: Any) -> ConformationSet:
        calls.append("find")
        return ConformationSet(
            conformations=conformations,
            tpt_result=tpt_result,
            metadata={"n_conformations": len(conformations)},
        )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        fake_find_conformations,
    )

    config = ConformationsConfig(topology_pdb=Path("app_intputs/test.pdb"))
    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is None
    assert result.tpt_summary["rate"] == pytest.approx(0.05)
    assert result.tpt_summary["n_pathways"] == 1
    assert result.metastable_states["0"]["n_states"] == 2
    assert result.transition_states[0]["committor"] == pytest.approx(0.5)
    assert result.plots.keys() == {"committors", "flux_network", "pathways"}
    assert "find" in calls
    assert any(key.startswith("load:") for key in calls)


@pytest.mark.unit
def test_conformations_config_controls_uncertainty_options(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    _patch_common_conformation_dependencies(monkeypatch, _workspace, _fake_dataset)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.85, 0.15], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    source_states = np.array([0])
    sink_states = np.array([1])
    pathways = [[0, 1]]
    captured_kwargs: Dict[str, Any] = {}

    def fake_find_conformations(**kwargs: Any) -> ConformationSet:
        captured_kwargs.update(kwargs)
        flux = np.array([[0.0, 0.2], [0.0, 0.0]])
        net_flux = np.array([[0.0, 0.2], [-0.2, 0.0]])
        tpt_result = TPTResult(
            source_states=source_states,
            sink_states=sink_states,
            forward_committor=np.array([0.0, 1.0]),
            backward_committor=np.array([1.0, 0.0]),
            flux_matrix=flux,
            net_flux=net_flux,
            total_flux=0.2,
            rate=0.05,
            mfpt=10.0,
            pathways=pathways,
            pathway_fluxes=np.array([0.2]),
            bottleneck_states=np.array([0]),
        )

        conformations = [
            Conformation(
                conformation_type="metastable",
                state_id=0,
                macrostate_id=0,
                frame_index=0,
                population=0.5,
                free_energy=1.0,
                metadata={"microstate_ids": [0, 1]},
                structure_path="metastable_0.pdb",
            ),
        ]

        return ConformationSet(
            conformations=conformations,
            tpt_result=tpt_result,
            metadata={"n_conformations": len(conformations)},
        )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        fake_find_conformations,
    )

    config = ConformationsConfig(
        topology_pdb=Path("app_intputs/test.pdb"),
        uncertainty_analysis=False,
        bootstrap_samples=25,
    )

    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is None
    assert captured_kwargs["uncertainty_analysis"] is False
    assert captured_kwargs["n_bootstrap"] == 25
    assert captured_kwargs["tica__dim"] == config.n_components
    assert captured_kwargs["committor_thresholds"] == tuple(config.committor_thresholds)


@pytest.mark.unit
def test_conformations_respects_tica_dimension(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    _patch_common_conformation_dependencies(
        monkeypatch, _workspace, _fake_dataset, patch_reduce=False
    )

    captured_reduce: Dict[str, Any] = {}

    def fake_reduce_features(
        features: Any, *, method: str, lag: int, n_components: int
    ) -> Any:
        captured_reduce.update(
            {"method": method, "lag": lag, "n_components": n_components}
        )
        return features

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.reduce_features",
        fake_reduce_features,
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.85, 0.15], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    source_states = np.array([0])
    sink_states = np.array([1])
    pathways = [[0, 1]]
    captured_kwargs: Dict[str, Any] = {}

    def fake_find_conformations(**kwargs: Any) -> ConformationSet:
        captured_kwargs.update(kwargs)
        flux = np.array([[0.0, 0.2], [0.0, 0.0]])
        net_flux = np.array([[0.0, 0.2], [-0.2, 0.0]])
        tpt_result = TPTResult(
            source_states=source_states,
            sink_states=sink_states,
            forward_committor=np.array([0.0, 1.0]),
            backward_committor=np.array([1.0, 0.0]),
            flux_matrix=flux,
            net_flux=net_flux,
            total_flux=0.2,
            rate=0.05,
            mfpt=10.0,
            pathways=pathways,
            pathway_fluxes=np.array([0.2]),
            bottleneck_states=np.array([0]),
        )

        conformations = [
            Conformation(
                conformation_type="metastable",
                state_id=0,
                macrostate_id=0,
                frame_index=0,
                population=0.5,
                free_energy=1.0,
                metadata={"microstate_ids": [0, 1]},
                structure_path="metastable_0.pdb",
            )
        ]

        return ConformationSet(
            conformations=conformations,
            tpt_result=tpt_result,
            metadata={"n_conformations": len(conformations)},
        )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        fake_find_conformations,
    )

    config = ConformationsConfig(
        topology_pdb=Path("app_intputs/test.pdb"),
        tica_dim=12,
        n_components=6,
    )

    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is None
    assert captured_reduce["method"] == "tica"
    assert captured_reduce["lag"] == config.lag
    assert captured_reduce["n_components"] == 12
    assert captured_kwargs["tica__dim"] == 12
    assert captured_kwargs["committor_thresholds"] == tuple(config.committor_thresholds)


@pytest.mark.unit
def test_conformations_uses_precomputed_deeptica_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    calls = _patch_common_conformation_dependencies(
        monkeypatch, _workspace, _fake_dataset, patch_reduce=False
    )

    def _raise_reduce(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("reduce_features should not be called for DeepTICA inputs")

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.reduce_features",
        _raise_reduce,
    )

    projection = np.linspace(0.0, 5.0, num=12, dtype=float).reshape(6, 2)
    projection_path = tmp_path / "deeptica_projection.npz"
    np.savez(projection_path, projection=projection)
    metadata = {
        "output_mean": [0.5, -0.5],
        "output_transform": [[1.0, 0.0], [0.0, 1.0]],
        "output_transform_applied": False,
    }
    metadata_path = tmp_path / "deeptica_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    captured: Dict[str, np.ndarray] = {}

    def fake_cluster(
        features_matrix: np.ndarray, *args: Any, **kwargs: Any
    ) -> SimpleNamespace:
        captured["matrix"] = np.asarray(features_matrix, dtype=float)
        return SimpleNamespace(labels=np.array([0, 1, 0, 1, 0, 1], dtype=int))

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.cluster_microstates",
        fake_cluster,
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        lambda **_kwargs: SimpleNamespace(
            tpt_result=SimpleNamespace(
                rate=0.1,
                mfpt=1.0,
                total_flux=0.2,
                pathways=[[0, 1]],
                source_states=np.array([0]),
                sink_states=np.array([1]),
            ),
            get_metastable_states=lambda: [],
            get_transition_states=lambda: [],
        ),
    )

    config = ConformationsConfig(
        topology_pdb=Path("app_intputs/test.pdb"),
        cv_method="deeptica",
        deeptica_projection_path=projection_path,
        deeptica_metadata_path=metadata_path,
    )

    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is None
    assert "plot" in calls
    expected = projection - np.asarray(metadata["output_mean"], dtype=float)
    np.testing.assert_allclose(captured["matrix"], expected)


@pytest.mark.unit
def test_conformations_deeptica_requires_projection_path(
    monkeypatch: pytest.MonkeyPatch,
    _workspace: WorkspaceLayout,
    _fake_dataset: Dict[str, Any],
) -> None:
    backend = WorkflowBackend(_workspace)
    shard_path = _workspace.shards_dir / "sample.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("{}", encoding="utf-8")

    _patch_common_conformation_dependencies(monkeypatch, _workspace, _fake_dataset)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_simple_msm",
        lambda *_args, **_kwargs: (
            np.array([[0.85, 0.15], [0.2, 0.8]], dtype=float),
            np.array([0.5, 0.5], dtype=float),
        ),
    )

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.find_conformations",
        lambda **_kwargs: None,
    )

    config = ConformationsConfig(
        topology_pdb=Path("app_intputs/test.pdb"),
        cv_method="deeptica",
        deeptica_projection_path=None,
    )

    result = backend.run_conformations_analysis([shard_path], config)

    assert result.error is not None
    assert "projection" in result.error.lower()
