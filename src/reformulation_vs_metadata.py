# Elisabeth

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os

# -------------------------
# Config
# -------------------------
PROCESSED_PATH = "NCTE_Transcripts/processed/ncte_single_utterances_cleaned.csv"
METADATA_PATH = "NCTE_Transcripts/transcript_metadata.csv"
OUT_DIR = "NCTE_Transcripts/processed/reformulation"

# Original metadata variables
METADATA_VARS = [
    "STATEVA_M",
    "CLAPS",
    "CLQF",
    "CLSTENG",
    "MLANG"
]

# Additional important metadata variables
IMPORTANT_VARS = [
    "SMQR",            # Student Mathematical Questioning & Reasoning
    "OSPMMR",          # Overall Student Participation in Meaning-Making & Reasoning
    "FORMAT_SMALLGRP"  # Small group indicator (binary)
]

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(PROCESSED_PATH)
meta = pd.read_csv(METADATA_PATH)

# -------------------------
# Keep teacher rows with valid overlap
# -------------------------
df_teacher = df[
    (df["is_teacher"] == True) &
    (df["teacher_reformulation_overlap"].notna())
].copy()

# -------------------------
# Aggregate to OBSID level
# -------------------------
agg = (
    df_teacher
    .groupby("OBSID")
    .agg(
        mean_reformulation_overlap=("teacher_reformulation_overlap", "mean"),
        n_teacher_turns=("teacher_reformulation_overlap", "count")
    )
    .reset_index()
)

# -------------------------
# Merge with metadata
# -------------------------
merged = agg.merge(meta, on="OBSID", how="left")

# -------------------------
# Make output directory
# -------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Analyze original variables
# -------------------------
available_vars = [v for v in METADATA_VARS if v in merged.columns]

print("Available original metadata variables:")
for v in available_vars:
    print(" -", v)

if not available_vars:
    raise ValueError("None of the requested metadata variables exist.")

# -------------------------
# Results storage
# -------------------------
results = []

def analyze_variable(sub, var, out_dir):
    """Run correlation, plot, and OLS regression for one variable."""
    if len(sub) < 10:
        print(f"Skipping {var}: not enough data")
        return None

    # Correlation: use point-biserial if binary
    if sub[var].nunique() == 2:
        r, p = stats.pointbiserialr(sub[var], sub["mean_reformulation_overlap"])
    else:
        r, p = stats.pearsonr(sub[var], sub["mean_reformulation_overlap"])

    print(f"\n{var}")
    print(f"  n = {len(sub)}")
    print(f"  r = {r:.3f}, p = {p:.4f}")

    # Scatter / boxplot
    plt.figure(figsize=(6,4))
    if sub[var].nunique() == 2:
        sns.boxplot(x=var, y="mean_reformulation_overlap", data=sub)
        plt.title(f"Teacher Reformulation vs {var} (binary)")
    else:
        sns.regplot(
            x=var,
            y="mean_reformulation_overlap",
            data=sub,
            scatter_kws={"alpha":0.6},
            line_kws={"color":"black"}
        )
        plt.title(f"Teacher Reformulation Overlap vs {var}")
    plt.xlabel(var)
    plt.ylabel("Mean Teacher Reformulation Overlap")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/reformulation_vs_{var}.png")
    plt.close()

    # OLS regression
    X = sm.add_constant(sub[var])
    y = sub["mean_reformulation_overlap"]
    model = sm.OLS(y, X).fit()

    with open(f"{out_dir}/ols_{var}.txt", "w") as f:
        f.write(model.summary().as_text())

    return {"variable": var, "n": len(sub), "pearson_r": r, "p_value": p}

# -------------------------
# Analyze original metadata variables
# -------------------------
for var in available_vars:
    sub = merged[["mean_reformulation_overlap", var]].dropna()
    res = analyze_variable(sub, var, OUT_DIR)
    if res:
        results.append(res)

# -------------------------
# Analyze additional important variables
# -------------------------
important_available = [v for v in IMPORTANT_VARS if v in merged.columns]

print("\nAvailable important metadata variables:")
for v in important_available:
    print(" -", v)

for var in important_available:
    sub = merged[["mean_reformulation_overlap", var]].dropna()
    res = analyze_variable(sub, var, OUT_DIR)
    if res:
        results.append(res)

# -------------------------
# Save summary table
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/reformulation_correlation_summary.csv", index=False)

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUT_DIR}")