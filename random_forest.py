import pickle
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix
)

# Load dataset
with open("qr_codes_29.pickle", "rb") as f:
    qr_codes = pickle.load(f)

with open("qr_codes_29_labels.pickle", "rb") as f:
    labels = pickle.load(f)

X = np.asarray(qr_codes).reshape(len(qr_codes), -1)
y = np.asarray(labels).ravel().astype(int)
row_ids = np.array([f"qr:{i}" for i in range(len(y))])

# Split: 70% train, 15% validation, 15% test
idx_train, idx_temp = train_test_split(
    np.arange(len(y)), test_size=0.30,
    random_state=42, stratify=y
)

idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50,
    random_state=42, stratify=y[idx_temp]
)

# Save split.csv
split_names = np.full(len(y), "", dtype=object)
split_names[idx_train] = "train"
split_names[idx_val] = "validation"
split_names[idx_test] = "test"

pd.DataFrame({
    "row_id": row_ids,
    "dataset": "quishing",
    "split": split_names,
    "label": y
}).to_csv("split.csv", index=False)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X[idx_train], y[idx_train])

# Choose threshold using validation set
val_scores = model.predict_proba(X[idx_val])[:, list(model.classes_).index(1)]
thresholds = np.unique(np.r_[0.0, val_scores, 1.0])

# Choose threshold with highest validation F1
from sklearn.metrics import f1_score
val_f1 = [
    f1_score(y[idx_val], (val_scores >= t).astype(int), zero_division=0)
    for t in thresholds
]
threshold = float(thresholds[int(np.argmax(val_f1))])

# Test scores
test_scores = model.predict_proba(X[idx_test])[:, list(model.classes_).index(1)]
test_labels = y[idx_test]
predictions = (test_scores >= threshold).astype(int)

pd.DataFrame({
    "row_id": row_ids[idx_test],
    "test_set": "quishing",
    "label": test_labels,
    "score": test_scores
}).to_csv("scores.csv", index=False)

print("Rows:", len(y))
print("Train / validation / test:", len(idx_train), len(idx_val), len(idx_test))
print("Validation-selected threshold:", threshold)
print("Test AUROC:", round(roc_auc_score(test_labels, test_scores), 4))
print("Test AP:", round(average_precision_score(test_labels, test_scores), 4))
print("Test precision:", round(precision_score(test_labels, predictions, zero_division=0), 4))
print("Test recall:", round(recall_score(test_labels, predictions, zero_division=0), 4))
print("Test confusion matrix:\n", confusion_matrix(test_labels, predictions))
print("scikit-learn version:", sklearn.__version__)
print("Saved split.csv and scores.csv")
