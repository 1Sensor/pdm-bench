import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)


class FeatureEngineeringSwitcher(BaseEstimator, TransformerMixin):
    """
    Transformer allowing dynamic switching between feature engineering methods.
    Supports: 'none', 'log', 'sqrt', 'poly'
    """

    def __init__(self, method: str = "none", degree: int | None = None):
        self.method = method
        self.degree = degree
        self.transformer_ = None

    def fit(self, x, y=None):
        if self.method == "none":
            self.transformer_ = FunctionTransformer(
                FeatureEngineeringSwitcher.identity_transform
            )
        elif self.method == "log":
            self.transformer_ = FunctionTransformer(
                FeatureEngineeringSwitcher.log_transform
            )
        elif self.method == "sqrt":
            self.transformer_ = FunctionTransformer(
                FeatureEngineeringSwitcher.sqrt_transform
            )
        elif self.method == "poly":
            if self.degree is None or self.degree < 1:
                raise ValueError("For 'poly' method, degree incorrectly specified.")
            self.transformer_ = PolynomialFeatures(degree=self.degree)
        else:
            raise ValueError(f"Unknown method: {self.method}.")
        return self.transformer_.fit(x, y)

    def transform(self, x):
        return self.transformer_.transform(x)

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def log_transform(x):
        return np.log1p(np.abs(x))

    @staticmethod
    def sqrt_transform(x):
        return np.sqrt(np.abs(x))


def create_pipeline(clf: BaseEstimator) -> Pipeline:
    """Create a pipeline with feature engineering, standard scaler and a cloned estimator."""
    return Pipeline(
        [
            (
                "fe",
                FeatureEngineeringSwitcher(),
            ),  # Feature Engineering as optional step
            ("scaler", StandardScaler()),
            ("estimator", clone(clf)),
        ]
    )
