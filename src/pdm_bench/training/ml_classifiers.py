# ruff: noqa: N806, N803, ERA001

import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


# isort: split
# Core ML utils
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


# isort: split
# ML models
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# isort: split
# Bayesian optimization
from skopt import BayesSearchCV

from pdm_bench.utils import create_pipeline


logger = logging.getLogger(__name__)


classifier_factory = {
    "KNN": lambda n_jobs=-1: KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs),
    "SVM": lambda **_: SVC(probability=True, random_state=42),
    "RF": lambda n_jobs=-1: RandomForestClassifier(random_state=42, n_jobs=n_jobs),
    "DT": lambda **_: DecisionTreeClassifier(random_state=42),
    "XGBoost": lambda n_jobs=-1: XGBClassifier(random_state=42, n_jobs=n_jobs),
    "GP": lambda **_: GaussianProcessClassifier(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)), random_state=42
    ),
    "AdaBoost": lambda **_: AdaBoostClassifier(random_state=42),
    "GaussianNB": lambda **_: GaussianNB(),
    "QDA": lambda **_: QuadraticDiscriminantAnalysis(),
    "LogisticRegression": lambda **_: LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=42
    ),
}

DEFAULT_BENCHMARK_SEARCH_SPACES = {
    "LogisticRegression": [
        {
            "fe__method": ["none", "log", "sqrt"],
            "estimator__solver": ["lbfgs"],
            "estimator__penalty": ["l2"],
            "estimator__C": (1e-3, 1e2, "log-uniform"),
        },
        {
            "fe__method": ["none", "log", "sqrt"],
            "estimator__solver": ["saga"],
            "estimator__penalty": ["l1"],
            "estimator__C": (1e-3, 1e2, "log-uniform"),
        },
    ],
    "SVM": [
        {
            "estimator__kernel": ["linear"],
            "estimator__C": (1e-4, 1e3, "log-uniform"),
        },
        # {
        #     "estimator__kernel": ["rbf"],
        #     "estimator__C": (1e-3, 1e3, "log-uniform"),
        #     "estimator__gamma": (1e-4, 1e1, "log-uniform"),
        # },
    ],
    "RF": {
        "estimator__n_estimators": (100, 500),
        "estimator__max_depth": (3, 30),
        "estimator__min_samples_split": (2, 10),
        "estimator__min_samples_leaf": (1, 10),
        "estimator__max_features": ["sqrt", "log2", None],
    },
    "XGBoost": {
        "estimator__n_estimators": (100, 500),
        "estimator__max_depth": (3, 10),
        "estimator__learning_rate": (0.01, 0.3, "log-uniform"),
        "estimator__subsample": (0.5, 1.0),
        "estimator__colsample_bytree": (0.5, 1.0),
        "estimator__min_child_weight": (1, 10),
        "estimator__reg_lambda": (1e-2, 1e2, "log-uniform"),
        "estimator__gamma": (0.0, 5.0),
    },
}


def train_ml_models(
    classifier_names: list,
    train_dataset: tuple,
    test_dataset: tuple,
    use_bayesian_search: bool = False,
    search_spaces: dict[str, object] | None = None,
    n_jobs: int = -1,
    bayes_n_iter: int = 10,
    bayes_cv: int = 3,
    bayes_n_points: int = 1,
    random_state: int = 42,
    save_path: str | None = None,
) -> dict:
    """
    Train machine learning models using provided train and test datasets.

    For each classifier name provided, the function creates a model instance,
    builds a pipeline, and trains it on the training data. If Bayesian hyperparameter
    search is enabled and a search space is provided for a model, it performs hyperparameter tuning.

    Args:
        classifier_names (list): List of classifier names.
        train_dataset (tuple): A tuple (X, y) where X contains the features and y the targets.
        test_dataset (tuple): A tuple (X, y) where X contains the features and y the targets.
        use_bayesian_search (bool, optional): Flag to use Bayesian search for hyperparameter tuning. Defaults to False.
        search_spaces (dict, optional): Dictionary mapping model names to their search spaces.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1 (using all processors).
        bayes_n_iter (int, optional): Number of iterations for Bayesian search. Defaults to 10.
        bayes_cv (int, optional): Number of cross-validation folds for Bayesian search. Defaults to 3.
        bayes_n_points (int, optional): Number of parameter settings evaluated per optimization
            iteration. Defaults to 1.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        save_path (str, optional): Path to save trained models. If None, models are not saved. Defaults to None.

    Returns:
        dict: A dictionary mapping each model name to a tuple:
            (trained_model, X_test, y_test)
    """
    classifiers = _get_classifiers(classifier_names, n_jobs)

    trained_models = {name: None for name in classifier_names}

    X_train, y_train = train_dataset
    X_test, y_test = test_dataset

    logger.info("Starting ML model training.")
    resolved_search_spaces = search_spaces or (
        DEFAULT_BENCHMARK_SEARCH_SPACES if use_bayesian_search else None
    )
    for name, clf in zip(classifier_names, classifiers, strict=True):
        logger.info("Training model %s", name)
        start_time = time.time()
        pipeline = create_pipeline(clf)

        if (
            use_bayesian_search
            and resolved_search_spaces is not None
            and name in resolved_search_spaces
        ):
            best_model = _train_with_bayes_cv(
                pipeline,
                X_train,
                y_train,
                resolved_search_spaces[name],
                bayes_cv,
                bayes_n_iter,
                bayes_n_points,
                n_jobs,
                random_state,
            )
        else:
            best_model = pipeline.fit(X_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time
        logger.info(
            "Model %s finished training in %.2f seconds with score %.4f.",
            name,
            train_time,
            best_model.score(X_test, y_test),
        )

        if save_path:
            path = Path(save_path)
            filename = (
                f"{name}_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            )
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, path / filename)

        trained_models[name] = best_model
        trained_models["test_data"] = X_test, y_test
    return trained_models


def _get_classifiers(classifier_names: list, n_jobs: int) -> list:
    """Get available classifiers from the classifier factory/"""
    for name in classifier_names:
        if name not in classifier_factory:
            raise ValueError(f"Not valid classifier name: {name}")

    return [classifier_factory[name](n_jobs=n_jobs) for name in classifier_names]


def _train_with_bayes_cv(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_space: dict,
    bayes_cv: int,
    bayes_n_iter: int,
    bayes_n_points: int,
    n_jobs: int,
    random_state: int,
) -> object:
    """
    Perform Bayesian hyperparameter search using BayesSearchCV.

    Args:
        pipeline (Pipeline): Pipeline to be optimized.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        search_space (dict): Search space defined for trained algorithm.
        bayes_cv (int): Number of cross-validation folds.
        bayes_n_iter (int): Number of iterations for Bayesian search.
        bayes_n_points (int): Number of parameter settings evaluated per search iteration.
        n_jobs (int): Number of parallel jobs used by the search.
        random_state (int): Random state for reproducibility.

    Returns:
        Any: Best estimator from the Bayesian search.
    """
    tuner = BayesSearchCV(
        pipeline,
        search_space,
        cv=bayes_cv,
        n_iter=bayes_n_iter,
        n_points=bayes_n_points,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=0,
    )
    tuner.fit(X_train, y_train)
    logger.info("Best BayesCV score: %.4f", tuner.best_score_)
    return tuner.best_estimator_
