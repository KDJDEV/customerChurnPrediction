from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)
