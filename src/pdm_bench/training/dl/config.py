from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OptimizerCfg:
    name: str = "adamw"  # "adam" | "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class SchedulerCfg:
    name: str = "exponential"  # "constant" | "exponential" | "plateau"
    gamma: float = 0.98
    factor: float = 0.5
    patience: int = 2


@dataclass
class TrainCfg:
    epochs: int = 10
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    label_smoothing: float = 0.05
    batch_size: int = 64
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    log_every: int = 0  # log every N batches; 0 = no logging
    log_train_metrics: bool = False  # log train acc/loss after each epoch
    class_weights: torch.Tensor | None = None
    random_state: int = 42
