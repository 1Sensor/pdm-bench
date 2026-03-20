# ruff: noqa: N806, N803

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

# Import functions to be tested from the ml_classifiers module
from pdm_bench.training.ml_classifiers import (
    DEFAULT_BENCHMARK_SEARCH_SPACES,
    _get_classifiers,
    train_ml_models,
)


# ------------------------------------------------------------------------------
# Test for the _get_classifiers function
# ------------------------------------------------------------------------------


def test_get_classifiers_valid():
    """
    Test _get_classifiers with valid classifier names.
    This test verifies that the function returns a list of estimators
    that are instances of scikit-learn's BaseEstimator.
    """
    valid_names = ["KNN", "SVM", "RF"]
    classifiers = _get_classifiers(valid_names, n_jobs=-1)
    # Check that we have the same number of classifiers as names
    assert len(classifiers) == len(valid_names)
    # Check that each classifier is a scikit-learn estimator
    for clf in classifiers:
        assert isinstance(clf, BaseEstimator)


def test_get_classifiers_invalid():
    """
    Test _get_classifiers with an invalid classifier name.
    This test expects a ValueError to be raised.
    """
    with pytest.raises(ValueError):
        _get_classifiers(["INVALID_NAME"], n_jobs=-1)


# ------------------------------------------------------------------------------
# Test for the train_ml_models function without Bayesian search
# ------------------------------------------------------------------------------


def test_train_ml_models_without_bayesian_search(tmp_path):
    """
    Test train_ml_models with use_bayesian_search=False.
    Uses a small dummy dataset to verify that the function returns a dictionary
    mapping classifier names to a tuple (trained_model, X_test, y_test).
    """
    # Create a small dummy dataset
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train = (X_train, y_train)
    test = (X_test, y_test)

    classifier_names = ["KNN", "SVM", "LogisticRegression"]

    # Call train_ml_models with Bayesian search disabled
    result = train_ml_models(
        classifier_names,
        train,
        test,
        use_bayesian_search=False,
        random_state=42,
        save_path=str(tmp_path),
    )

    # Check that the result is a dictionary with keys equal to classifier_names and consists of test data
    expected_keys = set(classifier_names)
    expected_keys.add("test_data")
    assert set(result.keys()) == expected_keys

    # Check that for each classifier, the returned tuple has three elements,
    # and that the trained model has a score method.
    for model_name, model in result.items():
        if model_name == "test_data":
            continue
        assert hasattr(model, "score")

    X_test, y_test = result["test_data"]
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # Check that for each classifier, a file exists in the save_path directory
    for clf_name in classifier_names:
        model_file = list(tmp_path.glob(f"{clf_name}_model_*.pkl"))
        assert model_file[0].exists(), f"File {model_file[0]} does not exist"
        # Optionally, load the file and check that the loaded model has a score method
        loaded_model = joblib.load(model_file[0])
        assert hasattr(loaded_model, "score"), f"Loaded model {clf_name} is invalid"


# ------------------------------------------------------------------------------
# Test for the train_ml_models function with Bayesian search enabled
# ------------------------------------------------------------------------------
def test_train_ml_models_with_bayesian_search(monkeypatch):
    """
    Test train_ml_models with use_bayesian_search=True.
    To avoid heavy computations, monkeypatch the _train_with_bayes_cv function
    to simply fit the pipeline normally.
    """

    # Define a dummy _train_with_bayes_cv function that just fits the pipeline.
    def dummy_train_with_bayes_cv(
        pipeline,
        X_train,
        y_train,
        name,
        bayes_cv,
        bayes_n_iter,
        bayes_n_points,
        n_jobs,
        random_state,
    ):
        return pipeline.fit(X_train, y_train)

    # Monkeypatch the _train_with_bayes_cv function in the ml_classifiers module
    monkeypatch.setattr(
        "pdm_bench.training.ml_classifiers._train_with_bayes_cv",
        dummy_train_with_bayes_cv,
    )

    # Create a small dummy dataset
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train = (X_train, y_train)
    test = (X_test, y_test)

    classifier_names = ["KNN", "RF"]

    # Define a dummy search_spaces dictionary
    search_spaces = {
        "KNN": {"estimator__n_neighbors": (1, 5)},
        "RF": {"estimator__n_estimators": (10, 50)},
    }

    result = train_ml_models(
        classifier_names,
        train,
        test,
        use_bayesian_search=True,
        search_spaces=search_spaces,
        bayes_n_iter=1,
        bayes_cv=2,
        bayes_n_points=1,
        random_state=42,
    )

    # Check that the result is a dictionary with correct keys and structure.
    expected_keys = set(classifier_names)
    expected_keys.add("test_data")
    assert set(result.keys()) == expected_keys

    for model_name, model in result.items():
        if model_name == "test_data":
            continue
        assert hasattr(model, "score")

    X_test, y_test = result["test_data"]
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_logistic_regression_benchmark_search_space_uses_fast_solver_specific_branches():
    search_spaces = DEFAULT_BENCHMARK_SEARCH_SPACES["LogisticRegression"]

    assert len(search_spaces) == 2

    lbfgs_space = next(
        space for space in search_spaces if space["estimator__solver"] == ["lbfgs"]
    )
    saga_space = next(
        space for space in search_spaces if space["estimator__solver"] == ["saga"]
    )

    assert lbfgs_space["fe__method"] == ["none", "log", "sqrt"]
    assert lbfgs_space["estimator__penalty"] == ["l2"]
    assert lbfgs_space["estimator__C"] == (1e-3, 1e2, "log-uniform")

    assert saga_space["fe__method"] == ["none", "log", "sqrt"]
    assert saga_space["estimator__penalty"] == ["l1"]
    assert saga_space["estimator__C"] == (1e-3, 1e2, "log-uniform")
