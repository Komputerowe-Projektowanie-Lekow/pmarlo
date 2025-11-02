import pytest
import torch

from pmarlo.features.data_loaders import make_loaders


def test_make_loaders_requires_minimum_samples():
    X = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="at least two samples"):
        make_loaders(X, X, batch_size=1, val_frac=0.5, num_workers=0)
