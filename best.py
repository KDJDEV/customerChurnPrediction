from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

matplotlib.use("Agg")

from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    make_preprocessor,
)
from src.models.train_model import create_pipeline, evaluate_model, split_data


RANDOM_SEED = 123
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"


def build_best_model():
    return create_pipeline(
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_dataset(DATA_DIR / "raw" / "train.csv")
    test_raw = load_dataset(DATA_DIR / "raw" / "test.csv")
    train = clean_dataset(train_raw, require_target=True)
    test = clean_dataset(test_raw, require_target=False)
    X = build_features(train)
    X_test = build_features(test)
    y = train["Churn"]

    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size=0.30, seed=RANDOM_SEED
    )

    best_model = build_best_model()
    best_model.fit(X_train, y_train)
    result, _, _ = evaluate_model(
        "Histogram gradient boosting", best_model, X_val, y_val
    )
    metrics_df = pd.DataFrame([result]).set_index("model")
    metrics_df.to_csv(REPORT_DIR / "best_model_validation.csv")

    best_model = build_best_model()
    best_model.fit(X, y)
    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame(
        {
            "CustomerID": test_raw["CustomerID"].astype(int),
            "Churn": np.clip(test_probabilities, 0.0, 1.0),
        }
    )
    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)