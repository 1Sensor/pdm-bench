from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

    from pdm_bench.training.dl.config import TrainCfg


def infer_input_shape(loader: DataLoader) -> tuple[int, ...]:
    """Infers input shape from a DataLoader."""
    xb, _ = next(iter(loader))
    if xb.dim() == 2:
        return (xb.size(1),)  # (D,)
    if xb.dim() == 3:
        return (xb.size(1), xb.size(2))  # (C, L)
    raise ValueError(f"Unsupported x shape: {tuple(xb.shape)}")


def evaluate(model: nn.Module, loader: DataLoader, device, criterion=None):
    """Evaluates model accuracy (and loss if given) on a DataLoader."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            if criterion is not None:
                total_loss += criterion(logits, yb).item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / max(1, total)
    loss = total_loss / max(1, len(loader)) if criterion is not None else None
    return acc, loss


def cfg_to_jsonable(cfg: TrainCfg) -> dict:
    """Safely converts TrainCfg to JSON-serializable dict (handles tensor fields)."""
    config_dict = asdict(cfg)
    class_weights = config_dict.get("class_weights")
    if isinstance(class_weights, torch.Tensor):
        config_dict["class_weights"] = class_weights.detach().cpu().tolist()
    return config_dict


def save_json(obj, path: Path):
    """Saves object as JSON to given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
