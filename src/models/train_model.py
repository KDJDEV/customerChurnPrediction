from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import make_preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.30,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def create_pipeline(
    classifier: BaseEstimator,
    preprocessor,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def select_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    max_depths: Sequence[int | None] = (3, 5, 8, 12, None),
) -> tuple[Pipeline, pd.DataFrame]:
    records = []
    models = {}
    for depth in max_depths:
        model = create_pipeline(
            DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=123,
            ),
            make_preprocessor(numeric_features, categorical_features),
        )
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        prediction = model.predict(X_val)
        records.append(
            {
                "max_depth": "none" if depth is None else depth,
                "accuracy": accuracy_score(y_val, prediction),
                "f1": f1_score(y_val, prediction, zero_division=0),
                "roc_auc": roc_auc_score(y_val, probability),
            }
        )
        models[depth] = model

    results = pd.DataFrame(records)
    best_index = int(results["roc_auc"].idxmax())
    best_depth = max_depths[best_index]
    return models[best_depth], results


def evaluate_model(
    name: str,
    model: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, object], pd.Series, pd.Series]:
    prediction = pd.Series(model.predict(X_val), index=y_val.index)
    probability = pd.Series(model.predict_proba(X_val)[:, 1], index=y_val.index)
    tn, fp, fn, tp = confusion_matrix(y_val, prediction).ravel()
    result = {
        "model": name,
        "accuracy": accuracy_score(y_val, prediction),
        "precision": precision_score(y_val, prediction, zero_division=0),
        "recall": recall_score(y_val, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_val, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probability),
    }
    return result, prediction, probability
