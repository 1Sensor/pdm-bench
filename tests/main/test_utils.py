import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pdm_tools.main.utils import create_pipeline


def test_create_pipeline_no_fe():
    from sklearn.neighbors import KNeighborsClassifier

    dummy_clf = KNeighborsClassifier()
    pipeline = create_pipeline(dummy_clf)
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "fe" in step_names
    assert "scaler" in step_names
    assert "estimator" in step_names


@pytest.mark.parametrize("method", ["log", "sqrt", "poly", "none", "log2"])
def test_create_pipeline_fe_methods(method):
    x = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)
    clf = LogisticRegression()
    pipeline = create_pipeline(clf)
    if method == "poly":
        pipeline.set_params(fe__method="poly", fe__degree=2)
    elif method == "log2":
        pipeline.set_params(fe__method=method)
        with pytest.raises(ValueError, match=f"Unknown method: {method}."):
            pipeline.fit(x, y)
        return
    else:
        pipeline.set_params(fe__method=method)
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    assert predictions.shape == y.shape


@pytest.mark.parametrize("degree", [None, 0])
def test_create_pipeline_fe_poly_invalid_degree(degree):
    x = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)
    clf = LogisticRegression()
    pipeline = create_pipeline(clf)
    pipeline.set_params(fe__method="poly", fe__degree=degree)
    with pytest.raises(
        ValueError, match="For 'poly' method, degree incorrectly specified."
    ):
        pipeline.fit(x, y)
