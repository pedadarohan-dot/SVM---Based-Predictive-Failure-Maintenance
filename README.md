# SVM-Based Predictive Failure Maintenance

A Soft Computing assignment on Support Vector Machines, applied to a real-world question: **can machine operating parameters distinguish normal operation from machine failure?**

## Overview

This project connects SVM theory to predictive maintenance using the AI4I predictive-maintenance dataset. The workflow follows:

`raw data → dataset understanding → feature selection → train/test split → scaling → SVM training → kernel comparison → evaluation → hyperparameter tuning → optimized-model evaluation`

## Dataset

- **Source:** AI4I predictive-maintenance dataset
- **Size:** 10,000 observations, 14 columns
- **Class distribution:** 96.61% no failure vs. 3.39% failure — a strongly imbalanced target

### Features used

Five continuous physical operating parameters were selected as a domain-informed baseline:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

`UDI` and `Product ID` were excluded as identifiers. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` were excluded because they are directly associated with failure and would introduce target leakage. `Type` was left out of the baseline in favor of focusing on the five continuous variables.

**Target:** `Machine failure` (0 = no failure, 1 = failure)

## Methodology

- **Train/test split:** 80/20 (8,000 / 2,000), stratified to preserve class proportions
- **Scaling:** Standardization (`z = (x − μ) / σ`), fit on training data only
- **Models compared:** Linear SVM, Polynomial SVM (degree 3), RBF SVM — all with `class_weight="balanced"`
- **Hyperparameter tuning:** Grid search with 5-fold cross-validation over `C` and `gamma` for the RBF kernel, optimized for F1 score
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrices — accuracy alone is misleading with a 3.39% failure rate

## Results

### Baseline kernel comparison

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Linear SVM     | 82.8%    | 14.25%    | 80.88% | 24.23%   |
| Polynomial SVM | 90.3%    | 25.20%    | 94.12% | 39.75%   |
| RBF SVM        | 91.3%    | 27.16%    | 92.65% | 42.00%   |

RBF gave the strongest baseline overall. Its confusion matrix was `[[1763, 169], [5, 63]]` — 63 of 68 actual failures detected (92.65% recall).

### Optimized RBF (best C = 100, best gamma = "scale")

| Metric    | Result |
|-----------|--------|
| Accuracy  | 95.2%  |
| Precision | 40.14% |
| Recall    | 83.82% |
| F1 Score  | 54.29% |

Confusion matrix: `[[1847, 85], [11, 57]]` — 57 of 68 actual failures detected.

Tuning improved accuracy, precision, and F1-score, but recall dropped from 92.65% to 83.82%. This is a genuine precision/recall trade-off, not an unqualified improvement — the tuned model catches slightly fewer failures while raising far fewer false alarms.

## What this project explored

- Feature selection requires reasoning about identifiers, physical relevance, and target leakage, not just correlation.
- Class imbalance makes raw accuracy an unreliable metric on its own.
- SVM is a maximum-margin classifier; kernels (linear, polynomial, RBF) let it model varying degrees of nonlinearity.
- `C` controls the penalty on classification errors; `gamma` controls how localized the RBF kernel's influence is.
- Hyperparameter tuning shifts the precision/recall trade-off — it doesn't uniformly improve every metric.
- Scikit-learn `Pipeline` parameter naming (`svc__C`, not `svc_C`) matters for `GridSearchCV`.

The same conceptual pipeline (dataset → feature selection → train/test split → standardization → SVM training → kernel comparison → prediction → evaluation) was also reproduced in MATLAB to confirm the methodology outside Python.

## Tech stack

- Python, pandas, NumPy
- scikit-learn (`SVC`, `GridSearchCV`, `Pipeline`, `StandardScaler`)
- Matplotlib

## Project structure

```
.
├── app.py              # Full pipeline: data loading, EDA, SVM training, tuning, visualizations
├── requirements.txt    # Python dependencies
└── README.md
```

## Running it

1. Download the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place `ai4i2020.csv` in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python app.py
   ```

The script prints dataset diagnostics and model metrics to the console, and displays five plots: class distribution, model comparison, the optimized model's confusion matrix, a false-positive/false-negative comparison, and a precision-recall scatter plot.

## Notes

This was built as a learning project for a Soft Computing course — the goal was to understand how SVM behaves on an imbalanced real-world classification problem, not to claim state-of-the-art results. The five selected features are a domain-informed baseline, not a claim of global feature optimality.
