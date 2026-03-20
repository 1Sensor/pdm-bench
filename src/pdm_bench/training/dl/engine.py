# ruff: noqa: N806, N803
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import ConcatDataset, DataLoader

from pdm_bench.training.dl.config import TrainCfg
from pdm_bench.training.dl.models import make_model
from pdm_bench.training.dl.utils import (
    cfg_to_jsonable,
    evaluate,
    infer_input_shape,
    save_json,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


torch.backends.cudnn.benchmark = True


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train_dl_models(
    model_names: Sequence[str],
    *,
    train_views: Sequence[torch.utils.data.Dataset],
    val_views: Sequence[torch.utils.data.Dataset] | None = None,
    test_views: Sequence[torch.utils.data.Dataset] | None = None,
    n_classes: int,
    cfg: TrainCfg,
    save_path: str | None = None,
):
    """Train multiple DL models on the benchmark dataset views."""
    torch.manual_seed(cfg.random_state)
    device = torch.device(cfg.device)
    use_pin_memory = device.type == "cuda"

    (
        (train_concat, val_concat, test_concat),
        (train_loader, val_loader, test_loader),
    ) = _prepare_dataloaders(train_views, val_views, test_views, cfg, use_pin_memory)

    input_shape = infer_input_shape(train_loader)

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for name in model_names:
        model = make_model(name, input_shape, n_classes).to(device)
        criterion, optimizer, scaler = _build_training_objects(model, cfg, device)
        scheduler = _build_scheduler(optimizer, cfg)
        metrics = {
            "epoch": [],
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
        }

        print(f"\n{_ts()} ▶️  Training {name} | device={device} | epochs={cfg.epochs}")
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            for i, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if scaler.is_enabled():
                    with amp.autocast(device_type=device.type, enabled=cfg.amp):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                running += loss.item()
                if cfg.log_every and (i + 1) % cfg.log_every == 0:
                    print(
                        f"{_ts()} [{name}] epoch {epoch} step {i + 1}/{len(train_loader)} "
                        f"loss {running / cfg.log_every:.4f}"
                    )
                    running = 0.0

            train_acc = train_loss = None
            val_acc = val_loss = None

            need_train_metrics = cfg.log_train_metrics or (
                cfg.scheduler.name == "plateau" and val_loader is None
            )
            if need_train_metrics:
                train_acc, train_loss = evaluate(model, train_loader, device, criterion)

            if val_loader is not None:
                val_acc, val_loss = evaluate(model, val_loader, device, criterion)

            _step_scheduler(scheduler, optimizer, cfg, val_loss, train_loss)

            metrics["epoch"].append(epoch)
            metrics["train_acc"].append(train_acc)
            metrics["train_loss"].append(train_loss)
            metrics["val_acc"].append(val_acc)
            metrics["val_loss"].append(val_loss)

            _log_epoch_line(
                name,
                epoch,
                cfg.epochs,
                train_acc,
                train_loss,
                val_acc,
                val_loss,
            )

        test_acc = test_loss = None
        if test_loader is not None:
            test_acc, test_loss = evaluate(model, test_loader, device, criterion)
            print(f"{_ts()} [{name}] TEST acc={test_acc:.4f} loss={test_loss:.4f}")

        print(f"{_ts()} ✅ {name} ready in {time.time() - t0:.2f}s")

        meta = {
            "model_name": name,
            "input_shape": tuple(input_shape),
            "n_classes": n_classes,
            "model_params": sum(p.numel() for p in model.parameters()),
            "epochs": cfg.epochs,
            "lr": cfg.optimizer.lr,
            "optimizer": cfg.optimizer.name,
            "weight_decay": cfg.optimizer.weight_decay,
            "scheduler": cfg.scheduler.name,
            "label_smoothing": cfg.label_smoothing,
            "batch_size": cfg.batch_size,
            "device": str(device),
            "amp": cfg.amp,
            "timestamp": timestamp,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_size": len(train_concat),
            "val_size": len(val_concat) if val_loader else 0,
            "test_size": len(test_concat) if test_loader else 0,
            "num_workers": cfg.num_workers,
            "random_state": cfg.random_state,
            "torch": torch.__version__,
        }

        _save_artifacts(save_path, name, model, meta, cfg, metrics, timestamp)

        results[name] = {
            "model": model,
            "input_shape": input_shape,
            "n_classes": n_classes,
            "metrics": {
                "val": {"acc": val_acc, "loss": val_loss},
                "test": {"acc": test_acc, "loss": test_loss},
            },
            "data": {
                "train_size": len(train_concat),
                "val_size": len(val_concat) if val_loader else 0,
                "test_size": len(test_concat) if test_loader else 0,
            },
        }

    return results


def _prepare_dataloaders(train_views, val_views, test_views, cfg, use_pin_memory):
    train_concat = ConcatDataset(list(train_views))
    val_concat = ConcatDataset(list(val_views)) if val_views else None
    test_concat = ConcatDataset(list(test_views)) if test_views else None

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(cfg.num_workers > 0),
        )

    train_loader = make_loader(train_concat, True)
    val_loader = (
        make_loader(val_concat, False) if val_concat and len(val_concat) > 0 else None
    )
    test_loader = (
        make_loader(test_concat, False)
        if test_concat and len(test_concat) > 0
        else None
    )

    _peek_loader("train", train_loader)
    if val_loader is not None:
        _peek_loader("val", val_loader)
    if test_loader is not None:
        _peek_loader("test", test_loader)

    return (train_concat, val_concat, test_concat), (
        train_loader,
        val_loader,
        test_loader,
    )


@torch.no_grad()
def _peek_loader(name: str, loader: torch.utils.data.DataLoader, max_ch: int = 4):
    try:
        xb, yb = next(iter(loader))
    except StopIteration:
        print(f"[peek/{name}] loader empty")
        return
    except Exception as exc:
        print(f"[peek/{name}] error while iterating:", repr(exc))
        return

    msg = [
        f"[peek/{name}] xb.shape={tuple(xb.shape)} xb.dtype={xb.dtype} "
        f"| yb.shape={tuple(yb.shape)} yb.dtype={yb.dtype}"
    ]
    if xb.dim() == 3:
        batch_size, channels, length = xb.shape
        mean = xb.mean(dim=(0, 2))
        std = xb.std(dim=(0, 2))
        msg.append(
            f"[peek/{name}] B={batch_size} C={channels} L={length} "
            f"| per-chan mean={mean[: min(channels, max_ch)].tolist()} "
            f"std={std[: min(channels, max_ch)].tolist()}"
        )
    elif xb.dim() == 2:
        batch_size, features = xb.shape
        msg.append(f"[peek/{name}] B={batch_size} D={features}")

    if yb.dtype == torch.long:
        classes, counts = torch.unique(yb, return_counts=True)
        msg.append(
            f"[peek/{name}] batch class dist: "
            + ", ".join(
                [
                    f"{int(label.item())}:{int(count.item())}"
                    for label, count in zip(classes, counts, strict=True)
                ]
            )
        )
    print("\n".join(msg))


def _build_training_objects(model, cfg, device):
    weight = None
    if cfg.class_weights is not None:
        weight = cfg.class_weights.detach().to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = amp.GradScaler(device=device.type, enabled=use_amp)
    return criterion, optimizer, scaler


def _build_scheduler(optimizer, cfg):
    if cfg.scheduler.name == "constant":
        return None
    if cfg.scheduler.name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.scheduler.gamma,
        )
    if cfg.scheduler.name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def _step_scheduler(scheduler, optimizer, cfg, val_loss, train_loss):
    if scheduler is None:
        return
    old_lr = optimizer.param_groups[0]["lr"]
    if cfg.scheduler.name == "plateau":
        metric = val_loss if val_loss is not None else train_loss
        if metric is None:
            return
        scheduler.step(metric)
    else:
        scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr < old_lr:
        print(f"{_ts()} [scheduler] LR reduced: {old_lr:.2e} -> {new_lr:.2e}")


def _log_epoch_line(name, epoch, epochs, train_acc, train_loss, val_acc, val_loss):
    msg = f"{_ts()} [{name}][epoch {epoch:3d}/{epochs}]"
    if train_acc is not None:
        msg += f"  |  train: acc={train_acc:6.3f} loss={train_loss:6.3f}"
    if val_acc is not None:
        msg += f"  |    val: acc={val_acc:6.3f} loss={val_loss:6.3f}"
    print(msg)


def _save_artifacts(save_path, name, model, meta, cfg, metrics, timestamp):
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = f"{name}_dl_model_{timestamp}.pt"

        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": meta,
            },
            Path(save_path) / fname,
        )

        config_path = Path(save_path) / f"{name}_config_{timestamp}.json"
        save_json(cfg_to_jsonable(cfg), config_path)

        metrics_path = Path(save_path) / f"{name}_metrics_{timestamp}.json"
        save_json(metrics, metrics_path)
