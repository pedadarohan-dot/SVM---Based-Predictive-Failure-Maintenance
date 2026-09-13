import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
file_path = "ai4i2020.csv"
df = pd.read_csv(file_path)
print("Dataset loaded successfully")
print("Data Shape: ", df.shape)

# Show Data
print("\n Columns Names: ")
print(df.columns.tolist())
print("\n First 5 rows: ")
print(df.head())

#Select Features
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# Features and Target
print("\n Selected Features: ")
print(X.head())

print("\n Target: ")
print(y.head())

# Check the Machine Failure Class Distribution
print("\n Machine Failure distribution: ")
print(y.value_counts())

print("\n Machine Failure Percentage: ")
print(y.value_counts(normalize=True) * 100)

# Checking whether there are any missing values or not
print("\n Missing Values: ")
print(df.isnull().sum())

# Train Test - Splitting the data in to training data and splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape: ")
print(X_train.shape)
print("Testing data shape: ")
print(X_test.shape)
print("\nTraining target distribution: ")
print(y_train.value_counts())
print("\nTesting Target Distribution: ")
print(y_test.value_counts())

# Standardization of the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nOriginal first training sample:")
print(X_train.iloc[0])

print("\nScaled first training sample:")
print(X_train_scaled[0])

# Linear SVM
svm_linear = SVC(
    kernel="linear",
    C=1.0,
    class_weight="balanced"
)
svm_linear.fit(X_train_scaled, y_train)
print("\nLinear SVM trained successfully")
y_pred = svm_linear.predict(X_test_scaled)
print("\nFirst 20 predictions: ")
print(y_pred[:20])
print("\nFirst 20 actual values: ")
print(y_test.iloc[:20].values)

# Evaluating all the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Linear SVM results")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1_score: ", f1)

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred, zero_division=0))

# Polynomial SVM
print("Polynomial SVM")
svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale",
    class_weight="balanced"
)
svm_poly.fit(X_train_scaled, y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
print("\nFirst 20 predictions: ")
print(y_pred[:20])
print("\nFirst 20 actual values: ")
print(y_test.iloc[:20].values)

# Evaluating all the metrics
print("Polynomial SVM results")
print("Accuracy: ", accuracy_score(y_test, y_pred_poly))
print("Precision: ", precision_score(y_test, y_pred_poly, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred_poly, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred_poly, zero_division=0))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_poly))
print("Classification Report: ")
print(classification_report(y_test, y_pred_poly, zero_division=0))

# RBF SVM
print("RBF SVM")
svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced"
)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print("\nFirst 20 predictions: ")
print(y_pred[:20])
print("\nFirst 20 actual values: ")
print(y_test.iloc[:20].values)

# Evaluating all the metrics
print("RBF SVM Metrics")
print("Accuracy: ", accuracy_score(y_test, y_pred_rbf))
print("Precision: ", precision_score(y_test, y_pred_rbf, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred_rbf, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred_rbf, zero_division=0))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_rbf))
print("Classification Report: ")
print(classification_report(y_test, y_pred_rbf, zero_division=0))

# Comparision Table
results = pd.DataFrame({
    "Model": [
        "Linear SVM",
        "Polynomial SVM",
        "RBF SVM"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_poly),
        accuracy_score(y_test, y_pred_rbf)
    ],

    "Precision": [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred_poly, zero_division=0),
        precision_score(y_test, y_pred_rbf, zero_division=0)
    ],

    "Recall": [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred_poly, zero_division=0),
        recall_score(y_test, y_pred_rbf, zero_division=0)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred_poly, zero_division=0),
        f1_score(y_test, y_pred_rbf, zero_division=0)

    ]
})
print("SVM Kernel Comparision")
print(results.to_string(index=False))

print("\n====================================")
print("CONFUSION MATRICES")
print("====================================")

print("\nLinear SVM:")
print(confusion_matrix(y_test, y_pred))

print("\nPolynomial SVM:")
print(confusion_matrix(y_test, y_pred_poly))

print("\nRBF SVM:")
print(confusion_matrix(y_test, y_pred_rbf))

print("\n====================================")
print("PREDICTED CLASS COUNTS")
print("====================================")

print("\nLinear SVM:")
print(pd.Series(y_pred).value_counts())

print("\nPolynomial SVM:")
print(pd.Series(y_pred_poly).value_counts())

print("\nRBF SVM:")
print(pd.Series(y_pred_rbf).value_counts())

# RBF SVM Hyperparameter Tuning - We are tuning the parameters C and gamma
print("RBF Hyper Parameter Tuning")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

# Parameters to test
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1]
}

# Grid search using 5-Fold cross validation
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Train
grid.fit(X_train, y_train)
print("Best Parameters: ")
print(grid.best_params_)
print("Best Cross-Validation Score: ")
print(grid.best_score_)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\n====================================")
print("OPTIMIZED RBF SVM - TEST RESULTS")
print("====================================")

print("Accuracy :",
      accuracy_score(y_test, y_pred_best))

print("Precision:",
      precision_score(y_test, y_pred_best, zero_division=0))

print("Recall   :",
      recall_score(y_test, y_pred_best, zero_division=0))

print("F1 Score :",
      f1_score(y_test, y_pred_best, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_best,
    zero_division=0
))

# ============================================================
# VISUALIZATION 1: MACHINE FAILURE DISTRIBUTION
# ============================================================

class_counts = df["Machine failure"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Normal Operation", "Machine Failure"],
    class_counts.values
)

plt.title("Machine Failure Class Distribution")
plt.ylabel("Number of Samples")

for bar, value in zip(bars, class_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 2: SVM MODEL COMPARISON
# ============================================================

models = [
    "Linear SVM",
    "Polynomial SVM",
    "RBF SVM",
    "Optimized RBF"
]

accuracy_values = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, y_pred_poly),
    accuracy_score(y_test, y_pred_rbf),
    accuracy_score(y_test, y_pred_best)
]

precision_values = [
    precision_score(y_test, y_pred, zero_division=0),
    precision_score(y_test, y_pred_poly, zero_division=0),
    precision_score(y_test, y_pred_rbf, zero_division=0),
    precision_score(y_test, y_pred_best, zero_division=0)
]

recall_values = [
    recall_score(y_test, y_pred, zero_division=0),
    recall_score(y_test, y_pred_poly, zero_division=0),
    recall_score(y_test, y_pred_rbf, zero_division=0),
    recall_score(y_test, y_pred_best, zero_division=0)
]

f1_values = [
    f1_score(y_test, y_pred, zero_division=0),
    f1_score(y_test, y_pred_poly, zero_division=0),
    f1_score(y_test, y_pred_rbf, zero_division=0),
    f1_score(y_test, y_pred_best, zero_division=0)
]

x = range(len(models))
width = 0.2

plt.figure(figsize=(11, 6))

plt.bar(
    [i - 1.5 * width for i in x],
    accuracy_values,
    width,
    label="Accuracy"
)

plt.bar(
    [i - 0.5 * width for i in x],
    precision_values,
    width,
    label="Precision"
)

plt.bar(
    [i + 0.5 * width for i in x],
    recall_values,
    width,
    label="Recall"
)

plt.bar(
    [i + 1.5 * width for i in x],
    f1_values,
    width,
    label="F1 Score"
)

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Comparison of SVM Models")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 3: FINAL CONFUSION MATRIX
# ============================================================

from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_best,
    display_labels=["Normal", "Failure"]
)

plt.title("Optimized RBF SVM - Confusion Matrix")

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 4: FALSE POSITIVE / FALSE NEGATIVE COMPARISON
# ============================================================

rbf_cm = confusion_matrix(y_test, y_pred_rbf)
optimized_cm = confusion_matrix(y_test, y_pred_best)

fp_rbf = rbf_cm[0, 1]
fn_rbf = rbf_cm[1, 0]

fp_optimized = optimized_cm[0, 1]
fn_optimized = optimized_cm[1, 0]

models = ["RBF SVM", "Optimized RBF"]

false_positives = [fp_rbf, fp_optimized]
false_negatives = [fn_rbf, fn_optimized]

x = range(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width/2 for i in x],
    false_positives,
    width,
    label="False Positives"
)

plt.bar(
    [i + width/2 for i in x],
    false_negatives,
    width,
    label="False Negatives"
)

plt.xticks(x, models)
plt.ylabel("Number of Samples")
plt.title("Effect of Hyperparameter Tuning on Prediction Errors")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 5: PRECISION-RECALL TRADE-OFF
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    precision_values,
    recall_values,
    s=100
)

for i, model in enumerate(models):
    plt.annotate(
        model,
        (precision_values[i], recall_values[i]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision-Recall Comparison of SVM Models")

plt.xlim(0, 0.5)
plt.ylim(0.75, 1.0)

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# FINAL RESULTS TABLE
# ============================================================

final_results = pd.DataFrame({
    "Model": [
        "Linear SVM",
        "Polynomial SVM",
        "RBF SVM",
        "Optimized RBF"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_poly),
        accuracy_score(y_test, y_pred_rbf),
        accuracy_score(y_test, y_pred_best)
    ],

    "Precision": [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred_poly, zero_division=0),
        precision_score(y_test, y_pred_rbf, zero_division=0),
        precision_score(y_test, y_pred_best, zero_division=0)
    ],

    "Recall": [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred_poly, zero_division=0),
        recall_score(y_test, y_pred_rbf, zero_division=0),
        recall_score(y_test, y_pred_best, zero_division=0)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred_poly, zero_division=0),
        f1_score(y_test, y_pred_rbf, zero_division=0),
        f1_score(y_test, y_pred_best, zero_division=0)
    ]
})

print("\n====================================")
print("FINAL SVM PERFORMANCE COMPARISON")
print("====================================")

print(final_results.round(4).to_string(index=False))