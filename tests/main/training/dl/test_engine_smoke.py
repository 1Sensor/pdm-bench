# tests/training/dl/test_engine_smoke.py
# ruff: noqa: N803
import json

import pytest
import torch
from torch.utils.data import Dataset

from pdm_tools.main.training.dl.config import OptimizerCfg, TrainCfg
from pdm_tools.main.training.dl.engine import train_dl_models


class ToyCnnView(Dataset):
    """Synthetic view CNN1D: (C, L) + label {0,1}."""

    def __init__(self, n=32, C=2, L=64, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, C, L, generator=g)
        self.y = (self.x.mean(dim=(1, 2)) > 0).long()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ToyMlpView(Dataset):
    """Synthetic view MLP: (D,) + label {0,1}."""

    def __init__(self, n=32, D=20, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, D, generator=g)
        self.y = (self.x.mean(dim=1) > 0).long()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def test_engine_runs_cnn1d_smoke(tmp_path):
    train_views = [ToyCnnView(n=32), ToyCnnView(n=32, seed=124)]
    test_views = [ToyCnnView(n=24, seed=125)]

    cfg = TrainCfg(
        epochs=1,
        optimizer=OptimizerCfg(lr=1e-3),
        batch_size=16,
        num_workers=0,
        log_train_metrics=False,
        amp=False,
        device="cpu",
    )

    out = train_dl_models(
        model_names=["cnn1d"],
        train_views=train_views,
        val_views=None,
        test_views=test_views,
        n_classes=2,
        cfg=cfg,
        save_path=str(tmp_path),
    )

    # --- output ---
    assert "cnn1d" in out
    res = out["cnn1d"]

    # key variables
    assert "model" in res and res["model"] is not None
    assert "input_shape" in res and isinstance(res["input_shape"], tuple)
    assert "n_classes" in res and isinstance(res["n_classes"], int)
    assert "metrics" in res and "test" in res["metrics"]

    # smoke: check if test metrics are present and of correct type
    acc = res["metrics"]["test"]["acc"]
    assert acc is None or isinstance(acc, float)
    loss = res["metrics"]["test"]["loss"]
    assert loss is None or isinstance(loss, float)

    # check if files were saved
    pt_files = list(tmp_path.glob("cnn1d_dl_model_*.pt"))
    cfg_files = list(tmp_path.glob("cnn1d_config_*.json"))
    met_files = list(tmp_path.glob("cnn1d_metrics_*.json"))
    assert len(pt_files) == 1
    assert len(cfg_files) == 1
    assert len(met_files) == 1

    # smoke: metrics file content is correct
    metrics = json.loads(met_files[0].read_text())
    for k in ["epoch", "train_acc", "train_loss", "val_acc", "val_loss"]:
        assert k in metrics


def test_engine_runs_mlp_smoke(tmp_path):
    # MLP expects (D,)
    train_views = [ToyMlpView(n=32), ToyMlpView(n=32, seed=999)]
    test_views = [ToyMlpView(n=24, seed=1000)]

    cfg = TrainCfg(
        epochs=1,
        optimizer=OptimizerCfg(lr=1e-3),
        batch_size=16,
        num_workers=0,
        log_train_metrics=False,
        amp=False,
        device="cpu",
    )

    out = train_dl_models(
        model_names=["mlp"],
        train_views=train_views,
        val_views=None,
        test_views=test_views,
        n_classes=2,
        cfg=cfg,
        save_path=str(tmp_path),
    )

    assert "mlp" in out
    res = out["mlp"]
    assert isinstance(res["input_shape"], tuple)
    assert isinstance(res["n_classes"], int)
    assert "metrics" in res and "test" in res["metrics"]
