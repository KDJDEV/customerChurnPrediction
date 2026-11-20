from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline

from src.features.build_features import make_preprocessor


def select_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    k_values: Sequence[int] = (3, 5, 9, 15, 25),
) -> tuple[Pipeline, pd.DataFrame]:
    candidates = sorted(set(k_values))
    if not candidates or candidates[0] < 1:
        raise ValueError("k values must be positive integers")
    if max(candidates) >= len(X_train):
        raise ValueError("k values must be smaller than the training set")

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    train_transformed = preprocessor.fit_transform(X_train, y_train)
    val_transformed = preprocessor.transform(X_val)
    neighbors = NearestNeighbors(
        n_neighbors=max(candidates), metric="euclidean", n_jobs=-1
    ).fit(train_transformed)
    _, indices = neighbors.kneighbors(val_transformed)
    train_target = y_train.to_numpy()

    records = []
    for k in candidates:
        probability = train_target[indices[:, :k]].mean(axis=1)
        prediction = (probability >= 0.5).astype(int)
        records.append(
            {
                "k": k,
                "accuracy": accuracy_score(y_val, prediction),
                "f1": f1_score(y_val, prediction, zero_division=0),
                "roc_auc": roc_auc_score(y_val, probability),
            }
        )

    results = pd.DataFrame(records)
    best_k = int(results.loc[results["roc_auc"].idxmax(), "k"])
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("classifier", KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)),
        ]
    )
    model.fit(X_train, y_train)
    return model, results
