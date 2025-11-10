import numpy as np

import pmarlo.transform.build as build_mod


def test_extract_cvs_selects_highest_variance_columns():
    dataset = {
        "X": np.array(
            [
                [0.0, -10.0, 0.0],
                [0.0, 10.0, -0.5],
                [0.0, -10.0, 0.5],
                [0.0, 10.0, 0.1],
            ],
            dtype=np.float64,
        ),
        "cv_names": ["const_col", "wide_spread", "moderate_spread"],
        "periodic": [False, True, False],
    }

    extracted = build_mod._extract_cvs(dataset)
    assert extracted is not None, "Expected CV pair to be extracted"
    cv1, cv2, names, periodic = extracted

    np.testing.assert_allclose(cv1, dataset["X"][:, 1])
    np.testing.assert_allclose(cv2, dataset["X"][:, 2])
    assert names == ("wide_spread", "moderate_spread")
    assert periodic == (True, False)
