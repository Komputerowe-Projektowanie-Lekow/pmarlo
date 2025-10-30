from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from pmarlo.features.deeptica.core.utils import safe_float
from pmarlo.utils.seed import set_global_seed


def test_set_global_seed_makes_rngs_reproducible():
    set_global_seed(1234)
    np_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_global_seed(1234)
    np_second = np.random.rand(3)
    torch_second = torch.rand(3)

    assert np.allclose(np_first, np_second)
    assert torch.allclose(torch_first, torch_second)


def test_safe_float_raises_for_bad_values():
    assert safe_float("3.14") == pytest.approx(3.14)
    with pytest.raises(ValueError, match="Cannot convert None to float"):
        safe_float(None)
    with pytest.raises(ValueError, match="Cannot convert {'oops': 'value'} to float"):
        safe_float({"oops": "value"})
