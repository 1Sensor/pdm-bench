import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pdm_bench.training.dl.models import (
    CNN1D,
    MLP,
    Cnn1dFft,
    Cnn2dStft,
    make_model,
)
from pdm_bench.training.dl.utils import infer_input_shape


class _ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_make_model_mlp_and_cnn1d_forward():
    mlp = make_model("mlp", (10,), n_classes=3)
    assert isinstance(mlp, MLP)
    out = mlp(torch.randn(4, 10))
    assert out.shape == (4, 3)

    cnn = make_model("cnn1d", (2, 16), n_classes=3)
    assert isinstance(cnn, CNN1D)
    out = cnn(torch.randn(4, 2, 16))
    assert out.shape == (4, 3)


def test_make_model_fft_and_stft_params():
    fft = make_model(
        "fft",
        (2, 32),
        n_classes=4,
        fft_log=False,
        fft_eps=1e-4,
        norm_eps=1e-3,
    )
    assert isinstance(fft, Cnn1dFft)
    assert fft.fe.log is False
    assert fft.fe.eps == 1e-4
    assert fft.inorm.eps == 1e-3
    out = fft(torch.randn(2, 2, 32))
    assert out.shape == (2, 4)

    stft = make_model(
        "stft",
        (2, 32),
        n_classes=4,
        stft_n_fft=16,
        stft_hop_length=4,
        stft_win_length=16,
        stft_window="hann",
        stft_log=False,
        stft_eps=1e-4,
        stft_center=False,
        norm_eps=1e-3,
    )
    assert isinstance(stft, Cnn2dStft)
    assert stft.fe.log is False
    assert stft.fe.eps == 1e-4
    assert stft.inorm.eps == 1e-3
    out = stft(torch.randn(2, 2, 32))
    assert out.shape == (2, 4)


def test_make_model_invalid_shape():
    with pytest.raises(ValueError, match="MLP expects input"):
        make_model("mlp", (2, 16), n_classes=3)


def test_infer_input_shape_from_mlp_batch():
    x = torch.randn(8, 5)
    y = torch.zeros(8, dtype=torch.long)
    ds = _ToyDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    assert infer_input_shape(loader) == (5,)


def test_infer_input_shape_from_cnn_batch():
    x = torch.randn(6, 3, 8)
    y = torch.zeros(6, 4)
    y[:, 2] = 1.0
    loader = DataLoader(_ToyDataset(x, y), batch_size=2)
    assert infer_input_shape(loader) == (3, 8)
