from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from pmarlo.results import FESResult


def test_result_roundtrip(tmp_path: Path) -> None:
    fes = FESResult(
        free_energy=np.zeros((2, 2)),
        xedges=np.array([0.0, 1.0, 2.0]),
        yedges=np.array([0.0, 1.0, 2.0]),
        cv1_name="x",
        cv2_name="y",
        temperature=300.0,
    )
    results = {"fes": fes}
    pkl = tmp_path / "analysis_results.pkl"
    js = tmp_path / "analysis_results.json"
    with pkl.open("wb") as fh:
        pickle.dump(results, fh)
    with js.open("w") as fh:
        json.dump({k: v.to_dict(metadata_only=True) for k, v in results.items()}, fh)

    loaded = pickle.load(pkl.open("rb"))
    assert isinstance(loaded["fes"], FESResult)
    assert loaded["fes"].output_shape == (2, 2)
    assert loaded["fes"].temperature == pytest.approx(300.0)

    meta = json.load(js.open())
    assert meta["fes"]["free_energy"]["shape"] == [2, 2]

    bad = fes.to_dict()
    bad["version"] = "0"
    with pytest.raises(ValueError):
        FESResult.from_dict(bad)
