"""
Task 1: Filter to agreed-upon annotations (exclude "REVIEW" rows)
"""

import pandas as pd
import os
print(os.getcwd())

# ── Load data ──
df = pd.read_csv("/Users/Jason/Desktop/Stanford/Winter_2026/CS 293/NCTE/GitHub/CS293/NCTE_Transcripts/processed/annotations/final_annotations_for_review.csv")
print(f"Total rows: {len(df)}")

# ── Label columns ──
label_cols = [
    "R1: References prior student content_final",
    "R2: Builds on student content_final",
    "R3: Invites further student thinking_final",
]

# ── Filter: keep rows where ALL three labels are 0 or 1 (no "REVIEW") ──
mask = True
for col in label_cols:
    mask = mask & (df[col].isin([0, 1, "0", "1"]))

df_agreed = df[mask].copy()

# Convert to int
for col in label_cols:
    df_agreed[col] = df_agreed[col].astype(int)

print(f"Agreed rows (no REVIEW on any R1/R2/R3): {len(df_agreed)}")


# ── Quick summary ──
print("=" * 60)
print('TASK 1: Filter to agreed-upon annotations (exclude "REVIEW" rows)')
print("=" * 60)

for col in label_cols:
    counts = df_agreed[col].value_counts().sort_index()
    print(f"\n{col}:")
    print(f"  0: {counts.get(0, 0)}")
    print(f"  1: {counts.get(1, 0)}")

# ── Save ──
df_agreed.to_csv("agreed_annotations.csv", index=False)
print(f"\nSaved to agreed_annotations.csv")


"""
Task 2: ChatGPT vs Human Accuracy Evaluation
=============================================
Merges human consensus annotations with ChatGPT predictions
and computes accuracy, precision, recall, F1, and Cohen's Kappa.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score
)

# ── Load data ──
human = pd.read_csv("/Users/Jason/Desktop/Stanford/Winter_2026/CS 293/NCTE/GitHub/CS293/agreed_annotations.csv")
gpt = pd.read_csv("/Users/Jason/Desktop/Stanford/Winter_2026/CS 293/NCTE/GitHub/CS293/chatgpt_annotations.csv")

# ── Merge on row ID ──
merged = human.merge(gpt, on="target_comb_idx", how="inner")
print(f"Merged rows: {len(merged)}")

# ── Variable pairs: (human_column, chatgpt_column) ──
variables = {
    "R1": ("R1: References prior student content_final", "R1_chatgpt"),
    "R2": ("R2: Builds on student content_final", "R2_chatgpt"),
    "R3": ("R3: Invites further student thinking_final", "R3_chatgpt"),
    "C1": ("C1. No student content available (N/A)_final", "C1_chatgpt"),
}

# ── Evaluate each variable ──
print("=" * 60)
print("TASK 2: ChatGPT vs Human Accuracy Evaluation")
print("=" * 60)

for var, (h_col, g_col) in variables.items():
    # Filter out any remaining REVIEW rows for this variable
    mask = merged[h_col].isin([0, 1, "0", "1"])
    subset = merged[mask].copy()
    y_true = subset[h_col].astype(int)
    y_pred = subset[g_col].astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n{var}:")
    print(f"  N = {len(y_true)}")
    print(f"  Accuracy:      {acc:.3f}")
    print(f"  Precision:     {prec:.3f}")
    print(f"  Recall:        {rec:.3f}")
    print(f"  F1 Score:      {f1:.3f}")
    print(f"  Cohen Kappa:   {kappa:.3f}")
    print(f"  Confusion Matrix (rows=human, cols=ChatGPT):")
    print(f"                Pred 0  Pred 1")
    print(f"    True 0    {cm[0][0]:>7d} {cm[0][1]:>7d}")
    print(f"    True 1    {cm[1][0]:>7d} {cm[1][1]:>7d}")