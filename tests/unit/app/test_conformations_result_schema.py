from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmarlo_webapp.app.backend.types import (
    ConformationsConfig,
    ConformationsResultSchema,
)


def test_conformations_result_schema_serializes_and_restores(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    pdb_path = Path("tests/_assets/3gd8-fixed.pdb").resolve()
    representative_path = output_dir / "state0.pdb"
    representative_path.write_text(pdb_path.read_text(encoding="utf-8"), encoding="utf-8")
    plot_path = output_dir / "pcca_states.png"
    plot_path.write_bytes(b"png")

    schema = ConformationsResultSchema(
        output_dir=output_dir,
        tpt={
            "rate": 0.012,
            "mfpt": 3.4,
            "total_flux": 1.8,
            "n_pathways": 2,
            "source_states": [0],
            "sink_states": [3],
            "tpt_converged": True,
            "pathway_iterations": 25,
            "pathway_max_iterations": 40,
        },
        metastable_states={
            "0": {
                "population": 0.55,
                "n_states": 3,
                "representative_pdb": representative_path,
            }
        },
        transition_states=[
            {
                "committor": 0.42,
                "state_index": 2,
                "representative_pdb": representative_path,
            }
        ],
        pathways=[[0, 1, 2, 3]],
        config=ConformationsConfig(topology_pdb=pdb_path),
        created_at="2025-02-21T12:00:00Z",
        plots={"pcca_states": plot_path},
        representative_pdbs=[representative_path],
        tpt_converged=True,
        tpt_pathway_iterations=25,
        tpt_pathway_max_iterations=40,
    )

    payload = json.loads(schema.model_dump_json())

    assert payload["output_dir"] == str(output_dir.resolve())
    assert payload["config"]["topology_pdb"] == str(pdb_path)
    assert payload["tpt"]["mfpt"] == pytest.approx(3.4)
    assert payload["metastable_states"]["0"]["representative_pdb"] == str(
        representative_path
    )

    result = schema.to_result()

    assert result.output_dir == output_dir
    assert result.tpt_summary["pathway_iterations"] == 25
    assert result.tpt_converged is True
    assert result.pathways == [[0, 1, 2, 3]]
    assert result.representative_pdbs[0].name == "state0.pdb"
    assert Path(result.metastable_states["0"]["representative_pdb"]).name == "state0.pdb"
