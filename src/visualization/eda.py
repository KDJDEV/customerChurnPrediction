from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    numeric = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Total Spend",
        "Last Interaction",
    ]
    categorical = ["Gender", "Subscription Type", "Contract Length"]

    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", ax=axis)
    axis.set_title("Churn Class Distribution")
    fig.tight_layout()
    fig.savefig(directory / "churn_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, column in zip(axes.flat, numeric):
        sns.boxplot(data=df, x="Churn", y=column, ax=axis)
        axis.set_title(column)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(directory / "numeric_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, column in zip(axes, categorical):
        sns.countplot(data=df, x=column, hue="Churn", ax=axis)
        axis.set_title(column)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(directory / "categorical_distributions.png", dpi=180)
    plt.close(fig)

    missing = df.isna().mean().sort_values(ascending=False)
    missing[missing > 0].rename("missing_fraction").to_csv(
        directory / "missing_values.csv"
    )
