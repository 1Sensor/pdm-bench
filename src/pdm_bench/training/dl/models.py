# ruff: noqa: N806, N803
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # (B, D)
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class FftFrontEnd(nn.Module):
    """rFFT over time; returns log amplitude: (B, C, L) -> (B, C, F)."""

    def __init__(self, log: bool = True, eps: float = 1e-6):
        super().__init__()
        self.log = log
        self.eps = eps

    def forward(self, x):  # x: (B, C, L), float32
        X = torch.fft.rfft(x, dim=-1)  # (B, C, F_complex)
        mag = X.abs()  # (B, C, F)
        if self.log:
            mag = mag.clamp_min(self.eps).log()  # log amplitude, numerically stable
        return mag


class StftFrontEnd(nn.Module):
    """STFT -> log-spectrogram: (B,C,L) -> (B,C,F,T)"""

    def __init__(
        self,
        n_fft=1024,
        hop_length=None,
        win_length=None,
        window="hann",
        log: bool = True,
        eps: float = 1e-6,
        center=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.win_length = win_length or n_fft
        self.center = center
        self.log = log
        self.eps = eps
        if window == "hann":
            self.register_buffer(
                "win", torch.hann_window(self.win_length), persistent=False
            )
        else:
            raise ValueError("Only hann supported in this snippet.")

    def forward(self, x):  # (B,C,L)
        B, C, L = x.shape
        # Vectorize across channels: (B,C,L) -> (B*C, L)
        x2 = x.reshape(B * C, L)

        win = self.win
        if win.device != x.device:
            win = win.to(device=x.device)

        # Optional AMP robustness: STFT is often most stable in float32.
        # Comment these 2 lines out if you know your torch+GPU handles fp16/bf16 STFT well.
        x2 = x2.float()
        win = win.float()

        S = torch.stft(
            x2,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            return_complex=True,
            center=self.center,
        )  # (B*C, F, T) complex
        S = S.abs()
        if self.log:
            S = S.clamp_min(self.eps).log()

        spec = S.reshape(B, C, S.shape[-2], S.shape[-1])  # (B,C,F,T)
        return spec


class Cnn1dFft(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        *,
        fft_log: bool = True,
        fft_eps: float = 1e-6,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.fe = FftFrontEnd(log=fft_log, eps=fft_eps)  # (B,C,L)->(B,C,F)
        self.inorm = nn.InstanceNorm1d(in_ch, affine=False, eps=norm_eps)
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B,C,L)
        x = self.fe(x)  # (B,C,F)
        x = self.inorm(x)  # (B, C, F) - stabilizes bands
        x = self.features(x)  # (B,128,1)
        return self.classifier(x)


class Cnn2dStft(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        *,
        stft_n_fft: int = 1024,
        stft_hop_length: int | None = None,
        stft_win_length: int | None = None,
        stft_window: str = "hann",
        stft_log: bool = True,
        stft_eps: float = 1e-6,
        stft_center: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.fe = StftFrontEnd(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            window=stft_window,
            log=stft_log,
            eps=stft_eps,
            center=stft_center,
        )
        self.inorm = nn.InstanceNorm2d(in_ch, affine=False, eps=norm_eps)
        self.features = nn.Sequential(  # input: (B, C, F, T)
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (F/2, T/2)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (F/4, T/4)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # (B,C,L)
        x = self.fe(x)  # (B,C,F,T)
        x = self.inorm(x)  # (B, C, F, T) - stabilizes TF dynamics
        x = self.features(x)  # (B,128,1,1)
        return self.classifier(x)


def make_model(
    name: str,
    input_shape: tuple[int, ...],
    n_classes: int,
    *,
    fft_log: bool = True,
    fft_eps: float = 1e-6,
    norm_eps: float = 1e-5,
    stft_n_fft: int = 1024,
    stft_hop_length: int | None = None,
    stft_win_length: int | None = None,
    stft_window: str = "hann",
    stft_log: bool = True,
    stft_eps: float = 1e-6,
    stft_center: bool = True,
) -> nn.Module:
    """Builds a model by name with optional FFT/STFT front-end parameters.

    Supported names:
        - "mlp"
        - "cnn", "cnn1d"
        - "cnn1d_fft", "fft"
        - "cnn2d_stft", "stft"

    Args:
        name: Model name (case-insensitive).
        input_shape: (D,) for MLP, or (C, L) for 1D models.
        n_classes: Number of output classes.
        fft_log: Whether to apply log to FFT magnitudes.
        fft_eps: Epsilon for FFT log stabilization.
        norm_eps: Epsilon for InstanceNorm layers.
        stft_n_fft: FFT size for STFT.
        stft_hop_length: Hop length for STFT (defaults to n_fft // 4).
        stft_win_length: Window length for STFT (defaults to n_fft).
        stft_window: Window type for STFT (currently "hann" only).
        stft_log: Whether to apply log to STFT magnitudes.
        stft_eps: Epsilon for STFT log stabilization.
        stft_center: Whether to pad and center STFT frames.
    """
    name = name.lower()
    if name == "mlp":
        if len(input_shape) != 1:
            raise ValueError(f"MLP expects input (D,), got {input_shape}.")
        return MLP(input_dim=input_shape[0], n_classes=n_classes)

    elif name in ("cnn", "cnn1d"):
        if len(input_shape) != 2:
            raise ValueError(f"CNN1D expects input (C, L), got {input_shape}.")
        return CNN1D(in_ch=input_shape[0], n_classes=n_classes)

    elif name in ("cnn1d_fft", "fft"):
        if len(input_shape) != 2:
            raise ValueError(f"CNN1D_FFT expects input (C, L), got {input_shape}.")
        return Cnn1dFft(
            in_ch=input_shape[0],
            n_classes=n_classes,
            fft_log=fft_log,
            fft_eps=fft_eps,
            norm_eps=norm_eps,
        )

    elif name in ("cnn2d_stft", "stft"):
        if len(input_shape) != 2:
            raise ValueError(f"CNN2D_STFT expects input (C, L), got {input_shape}.")
        return Cnn2dStft(
            in_ch=input_shape[0],
            n_classes=n_classes,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            stft_window=stft_window,
            stft_log=stft_log,
            stft_eps=stft_eps,
            stft_center=stft_center,
            norm_eps=norm_eps,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
