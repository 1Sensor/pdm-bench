from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import Dataset

from pdm_tools.main.evaluation.dl import evaluate_dl_classification_models


class _ToyView(Dataset):
    def __init__(self, xs: list[list[float]], ys: list[int]):
        self._xs = [torch.tensor(x, dtype=torch.float32) for x in xs]
        self._ys = [torch.tensor(y, dtype=torch.long) for y in ys]

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._xs[idx], self._ys[idx]


class _LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_evaluate_dl_models_returns_typed_results():
    views = [_ToyView(xs=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], ys=[0, 1, 1])]
    trained = {"cnn1d": {"model": _LinearModel()}}

    out = evaluate_dl_classification_models(
        trained,
        views,
        run_id="run-dl-001",
        split="test",
        pipeline="dl",
        device="cpu",
        batch_size=2,
        include_probabilities=True,
    )

    assert set(out.keys()) == {"cnn1d"}
    result = out["cnn1d"]
    assert result.predictions.model_name == "cnn1d"
    assert result.predictions.run_id == "run-dl-001"
    assert result.predictions.split == "test"
    assert result.predictions.pipeline == "dl"
    assert result.predictions.y_true == [0, 1, 1]
    assert result.predictions.y_pred == [0, 1, 0]
    assert result.predictions.y_score is not None
    assert result.summary.confusion_matrix == [[1, 0], [1, 1]]
    assert "accuracy" in result.summary.metrics
    assert "macro_f1" in result.summary.metrics


def test_evaluate_dl_models_skips_non_model_entries():
    views = [_ToyView(xs=[[1.0, 0.0]], ys=[0])]
    trained = {"not_model": {"model": object()}}

    out = evaluate_dl_classification_models(
        trained,
        views,
        run_id="run-dl-002",
    )

    assert out == {}


def test_evaluate_dl_models_persists_canonical_artifacts(tmp_path):
    views = [_ToyView(xs=[[1.0, 0.0], [0.0, 1.0]], ys=[0, 1])]
    trained = {"mlp": {"model": _LinearModel()}}

    out = evaluate_dl_classification_models(
        trained,
        views,
        run_id="run-dl-003",
        split="target_test",
        pipeline="da",
        artifacts_dir=tmp_path,
    )

    assert "mlp" in out
    pred_path = tmp_path / "mlp_target_test_predictions.json"
    summary_path = tmp_path / "mlp_target_test_summary.json"
    assert pred_path.exists()
    assert summary_path.exists()

    pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert pred_payload["run_id"] == "run-dl-003"
    assert pred_payload["pipeline"] == "da"
    assert summary_payload["split"] == "target_test"
    assert "accuracy" in summary_payload["metrics"]


def test_evaluate_dl_models_emits_view_ids_for_multi_view_eval():
    views = [
        _ToyView(xs=[[1.0, 0.0], [0.0, 1.0]], ys=[0, 1]),
        _ToyView(xs=[[1.0, 0.0]], ys=[1]),
    ]
    trained = {"cnn1d": {"model": _LinearModel()}}

    out = evaluate_dl_classification_models(
        trained,
        views,
        run_id="run-dl-004",
    )

    sample_ids = out["cnn1d"].predictions.sample_ids
    assert sample_ids is not None
    assert sample_ids["view_id"] == [0, 0, 1]


@pytest.mark.parametrize("eval_views", [None, []])
def test_evaluate_dl_models_handles_missing_or_empty_eval_views(eval_views):
    trained = {"cnn1d": {"model": _LinearModel()}}
    out = evaluate_dl_classification_models(
        trained,
        eval_views,
        run_id="run-dl-005",
    )
    assert out == {}
