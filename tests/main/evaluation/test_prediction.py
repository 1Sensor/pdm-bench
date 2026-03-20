from pdm_tools.main.evaluation.prediction import build_prediction_artifact


def test_build_prediction_artifact_converts_sequences_to_lists():
    artifact = build_prediction_artifact(
        task_type="classification",
        pipeline="ml",
        model_name="rf",
        run_id="run-001",
        split="test",
        y_true=(0, 1, 1),
        y_pred=(0, 1, 0),
        y_score=((0.9, 0.1), (0.1, 0.9), (0.7, 0.3)),
        labels=(0, 1),
        sample_ids={
            "recording_id": ("r1", "r1", "r2"),
            "window_index": (0, 1, 0),
        },
        metadata={"dataset": "toy"},
    )

    assert artifact.y_true == [0, 1, 1]
    assert artifact.y_pred == [0, 1, 0]
    assert artifact.y_score == [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]]
    assert artifact.labels == [0, 1]
    assert artifact.sample_ids == {
        "recording_id": ["r1", "r1", "r2"],
        "window_index": [0, 1, 0],
    }
    assert artifact.metadata["dataset"] == "toy"


def test_build_prediction_artifact_allows_optional_fields_to_be_absent():
    artifact = build_prediction_artifact(
        task_type="classification",
        pipeline="dl",
        model_name="cnn1d",
        run_id="run-002",
        split="target_test",
        y_true=[0, 1],
        y_pred=[0, 1],
    )

    assert artifact.y_score is None
    assert artifact.labels is None
    assert artifact.sample_ids is None
    assert artifact.metadata == {}
