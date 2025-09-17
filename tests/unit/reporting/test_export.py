import numpy as np

from pmarlo.reporting.export import write_conformations_csv_json


def test_write_conformations_csv_json(tmp_path):
    items = [{"id": 1, "array": np.array([1, 2]), "scalar": np.float64(1.2)}]
    write_conformations_csv_json(str(tmp_path), items)
    assert (tmp_path / "conformations_summary.csv").exists()
    assert (tmp_path / "states.json").exists()
