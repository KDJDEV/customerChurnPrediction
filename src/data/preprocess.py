import pandas as pd


REQUIRED_ID_COLUMN = "CustomerID"


def clean_dataset(
    df: pd.DataFrame, *, require_target: bool
) -> pd.DataFrame:
    required = {REQUIRED_ID_COLUMN}
    if require_target:
        required.add("Churn")

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[REQUIRED_ID_COLUMN].isna().any():
        raise ValueError("CustomerID contains missing values")
    if df.duplicated(REQUIRED_ID_COLUMN).any():
        raise ValueError("CustomerID must be unique")
    if require_target and not set(df["Churn"].unique()).issubset({0, 1}):
        raise ValueError("Churn must contain only 0 and 1")

    return df.copy()
