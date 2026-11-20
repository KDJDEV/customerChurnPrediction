import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.30,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
