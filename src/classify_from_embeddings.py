import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"""
@Jason:
To try MLP, change these lines:
from sklearn.neural_network import MLPClassifier
base_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)

To try decision tree, change these lines:
from sklearn.tree import DecisionTreeClassifier
base_model = DecisionTreeClassifier(max_depth=5)
"""

# =========================
# CONFIG
# =========================
TRAIN_EMBED_PATH = "synthetic_embeddings.npz"
VAL_EMBED_PATH = "human_embeddings.npz"

LABEL_NAMES = ["R1", "R2", "R3", "C1"]

# Per-label probability thresholds
# You should tune these on validation
THRESHOLDS = np.array([0.4, 0.4, 0.4, 0.3])

# =========================
# LOAD DATA
# =========================
print("Loading embeddings...")

train_data = np.load(TRAIN_EMBED_PATH)
X_train = train_data["embeddings"]
y_train = train_data["labels"]

val_data = np.load(VAL_EMBED_PATH)
X_val = val_data["embeddings"]  # pretty sure this is the line to change to concatenate heuristics @Evelyn
y_val = val_data["labels"]

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# =========================
# MODEL
# =========================
base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MultiOutputClassifier(base_model))
])

print("Training classifier...")
model.fit(X_train, y_train)

# =========================
# PROBABILITY PREDICTION
# =========================
print("\nEvaluating on human validation set...")

# Step 1: scale validation data
X_val_scaled = model.named_steps["scaler"].transform(X_val)

# Step 2: get classifier
clf = model.named_steps["clf"]

# Step 3: get probabilities per label
# Shape will be (num_samples, num_labels)
y_proba = np.column_stack([
    estimator.predict_proba(X_val_scaled)[:, 1]
    for estimator in clf.estimators_
])

# Step 4: apply thresholds
y_pred = (y_proba >= THRESHOLDS).astype(int)

# =========================
# EVALUATION
# =========================
per_label_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
micro_f1 = f1_score(y_val, y_pred, average="micro", zero_division=0)

for name, f1 in zip(LABEL_NAMES, per_label_f1):
    print(f"{name} F1: {f1:.4f}")

print(f"\nMacro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=LABEL_NAMES,
    zero_division=0
))

'''
Baseline:
Evaluating on human validation set...
R1 F1: 0.5202
R2 F1: 0.4948
R3 F1: 0.6769
C1 F1: 0.2532

Macro F1: 0.4863
Micro F1: 0.5331

Detailed Classification Report:
              precision    recall  f1-score   support

          R1       0.43      0.66      0.52        68
          R2       0.48      0.51      0.49        47
          R3       0.57      0.82      0.68        80
          C1       0.71      0.15      0.25        65

   micro avg       0.51      0.56      0.53       260
   macro avg       0.55      0.54      0.49       260
weighted avg       0.55      0.56      0.50       260
 samples avg       0.41      0.47      0.42       260
'''