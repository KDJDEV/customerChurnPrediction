import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBaselineClassifier(ClassifierMixin, BaseEstimator):
    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "RuleBaselineClassifier":
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        groups = self._risk_groups(X)
        target = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.group_probabilities_ = {}
        for group in range(4):
            mask = groups == group
            count = int(mask.sum())
            positives = int(target[mask].sum())
            self.group_probabilities_[group] = (positives + 1) / (count + 2)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "group_probabilities_"):
            raise RuntimeError("The baseline must be fitted before prediction")
        positive = np.array(
            [self.group_probabilities_[group] for group in self._risk_groups(X)]
        )
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _risk_groups(X: pd.DataFrame) -> np.ndarray:
        high_support = (
            pd.to_numeric(X["Support Calls"], errors="coerce").fillna(0) >= 5
        ).to_numpy(dtype=int)
        monthly = X["Contract Length"].eq("Monthly").to_numpy(dtype=int)
        return 2 * high_support + monthly
