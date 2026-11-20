from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def plot_model_results(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    probabilities: dict[str, pd.Series],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(predictions)

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        matrix = confusion_matrix(y_true, predictions[name])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for name in names:
        score = probabilities[name]
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Validation ROC Curves")
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "roc_curves.png", dpi=180)
    plt.close(fig)
