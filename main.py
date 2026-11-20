from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from src.models.dumb_model import RuleBaselineClassifier
from src.models.knn_model import select_knn_model
from src.models.train_model import (
    evaluate_model,
    select_decision_tree,
    split_data,
)
from src.visualization.eda import plot_eda
from src.visualization.performance import plot_model_results


RANDOM_SEED = 123
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_dataset(DATA_DIR / "raw" / "train.csv")
    train = clean_dataset(train_raw, require_target=True)

    plot_eda(train, REPORT_DIR / "eda")
    X = build_features(train)

    X_train, X_val, y_train, y_val = split_data(
        X, train["Churn"], test_size=0.30, seed=RANDOM_SEED
    )

    knn_model, k_results = select_knn_model(
        X_train,
        y_train,
        X_val,
        y_val,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        k_values=(3, 5, 9, 15, 25),
    )
    k_results.to_csv(REPORT_DIR / "knn_tuning.csv", index=False)
    best_k = int(knn_model.named_steps["classifier"].n_neighbors)

    baseline_model = RuleBaselineClassifier()
    baseline_model.fit(X_train, y_train)

    tree_model, tree_results = select_decision_tree(
        X_train,
        y_train,
        X_val,
        y_val,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        max_depths=(3, 5, 8, 12, None),
    )
    tree_results.to_csv(REPORT_DIR / "decision_tree_tuning.csv", index=False)
    best_depth = tree_model.named_steps["classifier"].max_depth

    models = {
        "Rule baseline": baseline_model,
        f"{best_k}-NN": knn_model,
        f"Decision tree (depth={best_depth})": tree_model,
    }

    metrics = []
    validation_predictions = {}
    validation_probabilities = {}
    for name, model in models.items():
        result, predictions, probabilities = evaluate_model(
            name, model, X_val, y_val
        )
        metrics.append(result)
        validation_predictions[name] = predictions
        validation_probabilities[name] = probabilities

    metrics_df = pd.DataFrame(metrics).set_index("model")
    metrics_df.to_csv(REPORT_DIR / "validation_metrics.csv")
    plot_model_results(
        y_val,
        validation_predictions,
        validation_probabilities,
        REPORT_DIR,
    )
