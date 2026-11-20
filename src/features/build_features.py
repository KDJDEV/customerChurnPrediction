from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Total Spend",
    "Last Interaction",
    "Payment Lag Days",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]
EXCLUDED_FEATURES = {
    "CustomerID",
    "Customer Status",
    "Payment Delay",
    "Churn",
    "Last Due Date",
    "Last Payment Date",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required_dates = {"Last Due Date", "Last Payment Date"}
    missing_dates = required_dates.difference(df.columns)
    if missing_dates:
        raise ValueError(f"Missing date columns: {sorted(missing_dates)}")

    result = df.copy()
    result["Support Calls"] = pd.to_numeric(
        result["Support Calls"].replace("none", 0), errors="coerce"
    )

    due_date = pd.to_datetime(
        "2000-" + result["Last Due Date"].astype(str), format="%Y-%m-%d"
    )
    payment_date = pd.to_datetime(
        "2000-" + result["Last Payment Date"].astype(str), format="%Y-%m-%d"
    )
    result["Payment Lag Days"] = (payment_date - due_date).dt.days

    return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        sparse_threshold=0.0,
    )
