# Customer Churn Prediction

For the best model for the submission I used a histogram gradient boosting model.

## Run Instructions

```text
conda create -n mat311-churn python=3.12
conda activate mat311-churn
pip install -r requirements.txt
```

Run the required baseline, k-NN, and decision tree evaluation:

```text
python main.py
```

Run the histogram gradient boosting model and create the submission:

```text
python best.py
```

The final leaderboard file will be saved to:

```text
submissions/submission.csv
```

## Generated Files

- `reports/eda/` — EDA figures and missing-value summary
- `reports/knn_tuning.csv` — metrics for each tested value of k
- `reports/decision_tree_tuning.csv` — tree-depth results
- `reports/validation_metrics.csv` — accuracy, precision, recall, specificity,
  F1, and ROC AUC for all three required models
- `reports/confusion_matrices.png` — three validation confusion matrices
- `reports/roc_curves.png` — all three validation ROC curves on one plot
- `reports/best_model_validation.csv` — validation metrics for the selected
  histogram gradient boosting model
- `reports/best_model_selection.csv` — model selected by validation ROC AUC
- `submissions/submission.csv` — final competition submission