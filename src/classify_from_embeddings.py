import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

'''
Make sure the 

To try MLP, change these lines:
from sklearn.neural_network import MLPClassifier
base_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)

To try descision tree, change these lines:
from sklearn.tree import DecisionTreeClassifier
base_model = DecisionTreeClassifier(max_depth=5)
'''

# =========================
# CONFIG
# =========================
TRAIN_EMBED_PATH = "synthetic_embeddings.npz"
VAL_EMBED_PATH = "human_embeddings.npz"

LABEL_NAMES = ["R1", "R2", "R3", "C1"]

# =========================
# LOAD DATA
# =========================
print("Loading embeddings...")

train_data = np.load(TRAIN_EMBED_PATH)
X_train = train_data["embeddings"]
y_train = train_data["labels"]

val_data = np.load(VAL_EMBED_PATH)
X_val = val_data["embeddings"]
y_val = val_data["labels"]

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# =========================
# MODEL
# =========================
# Scaling helps logistic regression
base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"  # helpful if labels are imbalanced
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MultiOutputClassifier(base_model))
])

print("Training classifier...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("\nEvaluating on human validation set...")

y_pred = model.predict(X_val)

# Per-label F1
per_label_f1 = f1_score(y_val, y_pred, average=None)
macro_f1 = f1_score(y_val, y_pred, average="macro")
micro_f1 = f1_score(y_val, y_pred, average="micro")

for name, f1 in zip(LABEL_NAMES, per_label_f1):
    print(f"{name} F1: {f1:.4f}")

print(f"\nMacro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

# Optional detailed report
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=LABEL_NAMES))