from __future__ import annotations

import sys
import types
from pathlib import Path
import importlib
from typing import Iterator

import numpy as np
import pytest
import torch
import torch.nn as nn


if "mlcolvar" not in sys.modules:
    _mlc = types.ModuleType("mlcolvar")
    _cvs = types.ModuleType("mlcolvar.cvs")
    _utils = types.ModuleType("mlcolvar.utils")
    _timelagged = types.ModuleType("mlcolvar.utils.timelagged")
    _mlc.cvs = _cvs
    _mlc.utils = _utils
    _utils.timelagged = _timelagged
    _timelagged.create_timelagged_dataset = lambda *args, **kwargs: None
    _cvs.DeepTICA = object
    sys.modules["mlcolvar"] = _mlc
    sys.modules["mlcolvar.cvs"] = _cvs
    sys.modules["mlcolvar.utils"] = _utils
    sys.modules["mlcolvar.utils.timelagged"] = _timelagged


from pmarlo.features.deeptica.losses import VAMP2Loss

class _DummyDictDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, data_lag: torch.Tensor):
        self._data = data
        self._lag = data_lag

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __getitem__(self, idx: int):
        return {
            "data": self._data[idx],
            "data_lag": self._lag[idx],
        }


def _install_stub_mlcolvar():
    modules_to_restore = {}
    for name in [
        "mlcolvar",
        "mlcolvar.cvs",
        "mlcolvar.utils",
        "mlcolvar.utils.timelagged",
        "mlcolvar.data",
    ]:
        modules_to_restore[name] = sys.modules.get(name)

    mlcolvar = types.ModuleType("mlcolvar")
    cvs = types.ModuleType("mlcolvar.cvs")
    utils = types.ModuleType("mlcolvar.utils")
    timelagged = types.ModuleType("mlcolvar.utils.timelagged")
    data_mod = types.ModuleType("mlcolvar.data")

    class DummyDeepTICA(nn.Module):
        def __init__(self, layers, n_cvs, activation: str = "tanh", options=None):
            super().__init__()
            options = options or {}
            activation = (activation or options.get("activation", "gelu")).lower()
            if activation == "tanh":
                act_cls = nn.Tanh
            elif activation == "relu":
                act_cls = nn.ReLU
            else:
                act_cls = nn.GELU
            modules = []
            for i in range(len(layers) - 1):
                modules.append(nn.Linear(layers[i], layers[i + 1]))
                if i < len(layers) - 2:
                    modules.append(act_cls())
            self.model = nn.Sequential(*modules)

        def forward(self, x):
            return self.model(x)

        def fit(
            self,
            dataset,
            batch_size: int = 32,
            max_epochs: int = 10,
            early_stopping_patience: int = 5,
            shuffle: bool = False,
            **_,
        ) -> "DummyDeepTICA":
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            opt = torch.optim.Adam(self.parameters(), lr=1e-3)
            loss_fn = VAMP2Loss()
            for _ in range(min(max_epochs, 5)):
                for batch in loader:
                    x_t = batch["data"].to(dtype=torch.float32)
                    x_tau = batch["data_lag"].to(dtype=torch.float32)
                    opt.zero_grad()
                    loss, _ = loss_fn(self(x_t), self(x_tau))
                    loss.backward()
                    opt.step()
            return self

    class DictModule:
        def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True, split: dict[str, float] | None = None, num_workers: int = 0):
            if split is None:
                split = {"train": 0.9, "val": 0.1}
            n = len(dataset)
            n_train = max(1, int(split.get("train", 0.9) * n))
            indices = np.arange(n)
            self.train_subset = torch.utils.data.Subset(dataset, indices[:n_train])
            if n_train < n:
                self.val_subset = torch.utils.data.Subset(dataset, indices[n_train:])
            else:
                self.val_subset = torch.utils.data.Subset(dataset, indices[-1:])
            self.batch_size = batch_size
            self.shuffle = shuffle
        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def create_timelagged_dataset(data: np.ndarray, lag: int) -> _DummyDictDataset:
        data = torch.as_tensor(data, dtype=torch.float32)
        return _DummyDictDataset(data[:-lag], data[lag:])

    mlcolvar.cvs = cvs
    cvs.DeepTICA = DummyDeepTICA
    mlcolvar.utils = utils
    utils.timelagged = timelagged
    timelagged.create_timelagged_dataset = create_timelagged_dataset
    mlcolvar.data = data_mod
    data_mod.DictModule = DictModule

    sys.modules["mlcolvar"] = mlcolvar
    sys.modules["mlcolvar.cvs"] = cvs
    sys.modules["mlcolvar.utils"] = utils
    sys.modules["mlcolvar.utils.timelagged"] = timelagged
    sys.modules["mlcolvar.data"] = data_mod

    def cleanup():
        for name, module in modules_to_restore.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return cleanup


def _install_stub_lightning():
    modules_to_restore = {}
    for name in [
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.loggers",
    ]:
        modules_to_restore[name] = sys.modules.get(name)

    pl = types.ModuleType("pytorch_lightning")
    callbacks_mod = types.ModuleType("pytorch_lightning.callbacks")
    loggers_mod = types.ModuleType("pytorch_lightning.loggers")

    class LightningModule(nn.Module):
        def __init__(self):
            super().__init__()
            self._logged_metrics = {}

        def save_hyperparameters(self, params):
            self.hparams = types.SimpleNamespace(**params)

        def log(self, name, value, **_):
            if isinstance(value, torch.Tensor):
                value = value.detach()
            self._logged_metrics[name] = value

    class Trainer:
        def __init__(
            self,
            max_epochs: int,
            min_epochs: int = 1,
            enable_progress_bar: bool = False,
            logger=False,
            callbacks=None,
            **_,
        ):
            self.max_epochs = max_epochs
            self.min_epochs = min_epochs
            self.callbacks = callbacks or []
            self.callback_metrics = {}

        def fit(self, model, datamodule=None, train_dataloaders=None, val_dataloaders=None):
            if not hasattr(model, "device"):
                model.device = torch.device("cpu")
            model = model.to(torch.device("cpu"))
            if datamodule is not None:
                train_loader = datamodule.train_dataloader()
                val_loader = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None
            else:
                train_loader = train_dataloaders
                val_loader = val_dataloaders

            opt_conf = model.configure_optimizers()
            if isinstance(opt_conf, dict):
                optimizer = opt_conf["optimizer"]
                scheduler = opt_conf.get("lr_scheduler")
                if isinstance(scheduler, dict):
                    scheduler = scheduler.get("scheduler")
            else:
                optimizer = opt_conf
                scheduler = None
            epochs = max(self.min_epochs, self.max_epochs)
            for _ in range(epochs):
                if hasattr(model, "on_train_epoch_start"):
                    model.on_train_epoch_start()
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = model.training_step(batch, 0)
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    loss.backward()
                    if hasattr(model, "on_after_backward"):
                        model.on_after_backward()
                    optimizer.step()
                if scheduler is not None and hasattr(scheduler, "step"):
                    scheduler.step()
                self.callback_metrics = {
                    name: (val.detach().cpu() if isinstance(val, torch.Tensor) else torch.tensor(float(val)))
                    for name, val in model._logged_metrics.items()
                }
                for cb in self.callbacks:
                    if hasattr(cb, "on_train_epoch_end"):
                        cb.on_train_epoch_end(self, model)
                if hasattr(model, "on_train_epoch_end"):
                    model.on_train_epoch_end()
                if val_loader is not None:
                    if hasattr(model, "on_validation_epoch_start"):
                        model.on_validation_epoch_start()
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            model.validation_step(batch, 0)
                    self.callback_metrics = {
                        name: (val.detach().cpu() if isinstance(val, torch.Tensor) else torch.tensor(float(val)))
                        for name, val in model._logged_metrics.items()
                    }
                    for cb in self.callbacks:
                        if hasattr(cb, "on_validation_epoch_end"):
                            cb.on_validation_epoch_end(self, model)
                    if hasattr(model, "on_validation_epoch_end"):
                        model.on_validation_epoch_end()
                model._logged_metrics = {}

    class Callback:
        pass

    class EarlyStopping(Callback):
        def __init__(self, *args, **kwargs):
            pass

    class ModelCheckpoint(Callback):
        def __init__(self, dirpath=None, filename=None, monitor=None, mode="max", save_top_k=1, save_last=False, every_n_epochs=None):
            self.dirpath = dirpath
            self.filename = filename
            self.monitor = monitor
            self.mode = mode
            self.best_model_path = None

    class CSVLogger:
        def __init__(self, save_dir: str, name: str, version: str):
            self.log_dir = Path(save_dir) / name / str(version)
            self.log_dir.mkdir(parents=True, exist_ok=True)

    class TensorBoardLogger:
        def __init__(self, save_dir: str, name: str):
            self.log_dir = Path(save_dir) / name
            self.log_dir.mkdir(parents=True, exist_ok=True)

    pl.Trainer = Trainer
    pl.LightningModule = LightningModule
    callbacks_mod.Callback = Callback
    callbacks_mod.EarlyStopping = EarlyStopping
    callbacks_mod.ModelCheckpoint = ModelCheckpoint
    loggers_mod.CSVLogger = CSVLogger
    loggers_mod.TensorBoardLogger = TensorBoardLogger

    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks_mod
    sys.modules["pytorch_lightning.loggers"] = loggers_mod

    def cleanup():
        for name, module in modules_to_restore.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return cleanup


@pytest.fixture
def deeptica_module(monkeypatch):
    restore_ml = _install_stub_mlcolvar()
    restore_lightning = _install_stub_lightning()
    original_dataloader = torch.utils.data.DataLoader
    def _safe_dataloader(*args, **kwargs):

        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
        kwargs["num_workers"] = 0
        return original_dataloader(*args, **kwargs)
    torch.utils.data.DataLoader = _safe_dataloader
    sys.modules.pop("pmarlo.features.deeptica", None)
    import pmarlo.features.deeptica as deeptica

    yield deeptica

    torch.utils.data.DataLoader = original_dataloader
    restore_lightning()
    restore_ml()
    sys.modules.pop("pmarlo.features.deeptica", None)
    importlib.invalidate_caches()


def generate_pairs(n_frames: int = 200, lag: int = 1, seed: int = 1):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_frames + lag, 2)).astype(np.float32)
    idx = np.arange(0, n_frames, dtype=np.int64)
    return [data[:-lag], data[lag:]], (idx, idx + lag)


def _flatten_history_values(history, keys: Iterator[str]):
    for key in keys:
        values = history.get(key)
        if isinstance(values, list) and values:
            array = np.asarray(values, dtype=float)
            assert np.isfinite(array).all(), f"Non-finite values in {key}"


def test_training_history_curves_are_finite(deeptica_module):
    X_pair, pairs = generate_pairs()
    X_list = [X_pair[0], X_pair[1]]
    cfg = deeptica_module.DeepTICAConfig(
        lag=1,
        n_out=2,
        max_epochs=4,
        early_stopping=2,
        batch_size=32,
        hidden=(32, 16),
        num_workers=0,
        linear_head=False,
    )
    model = deeptica_module.train_deeptica(X_list, pairs, cfg, weights=None)
    history = model.training_history

    loss_curve = history.get("loss_curve") or []
    objective_curve = history.get("objective_curve") or []
    patience = getattr(cfg, "early_stopping", 0)
    assert len(objective_curve) >= max(1, len(loss_curve) - patience)

    _flatten_history_values(
        history,
        (
            "loss_curve",
            "objective_curve",
            "val_score_curve",
            "val_score",
            "var_z0_curve",
            "var_zt_curve",
            "cond_c00_curve",
            "cond_ctt_curve",
            "grad_norm_curve",
        ),
    )
    output_variance = history.get("output_variance")
    if output_variance is not None:
        assert np.isfinite(np.asarray(output_variance, dtype=float)).all()
    assert history.get("grad_norm_curve"), "Expected grad_norm_curve to be populated"
