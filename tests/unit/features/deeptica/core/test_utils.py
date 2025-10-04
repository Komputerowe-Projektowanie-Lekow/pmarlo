from __future__ import annotations

import numpy as np
import pytest
import torch

from pmarlo.features.deeptica.core.utils import safe_float, set_all_seeds


def test_set_all_seeds_makes_rngs_reproducible():
    set_all_seeds(1234)
    np_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_all_seeds(1234)
    np_second = np.random.rand(3)
    torch_second = torch.rand(3)

    assert np.allclose(np_first, np_second)
    assert torch.allclose(torch_first, torch_second)


def test_safe_float_handles_bad_values():
    assert safe_float("3.14") == pytest.approx(3.14)
    assert safe_float({"oops": "value"}, default=1.5) == pytest.approx(1.5)
