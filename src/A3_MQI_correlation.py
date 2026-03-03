import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------
# 1. Load data
# -----------------------------
annotations = pd.read_csv(
    "NCTE_Transcripts/processed/annotations/final_combined_ALLannotation.csv"
)
metadata = pd.read_csv(
    "NCTE_Transcripts/transcript_metadata.csv"
)

# -----------------------------
# 2. Exclude C1 = 1 rows
# -----------------------------
annotations_filtered = annotations[annotations["C1"] != 1]

# -----------------------------
# 3. Aggregate utterance → classroom level
# -----------------------------
resp_cols = ["R1", "R2", "R3"]

classroom_resp = (
    annotations_filtered
    .groupby("OBSID")[resp_cols]
    .mean()
    .reset_index()
)

# Total Responsiveness = mean of R1, R2, R3
classroom_resp["Total_Responsiveness"] = classroom_resp[resp_cols].mean(axis=1)

# -----------------------------
# 4. Merge with metadata
# -----------------------------
merged = classroom_resp.merge(
    metadata[["OBSID", "MLANG", "SMQR", "OSPMMR"]],
    on="OBSID",
    how="left"
)

resp_vars = ["R1", "R2", "R3", "Total_Responsiveness"]
mqi_vars = ["MLANG", "SMQR", "OSPMMR"]

# -----------------------------
# 5. Generate plots
# -----------------------------
for mqi in mqi_vars:

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, resp in enumerate(resp_vars):

        df_clean = merged[[resp, mqi]].dropna()

        if len(df_clean) < 3:
            continue

        x = df_clean[resp]
        y = df_clean[mqi]

        axes[i].scatter(x, y, color="gray", s=20, alpha=0.6)

        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = slope * x_vals + intercept
        axes[i].plot(x_vals, y_vals)

        r, p = pearsonr(x, y)

        axes[i].set_xlabel(resp)
        axes[i].set_ylabel(mqi)
        axes[i].set_title(f"{resp}\nr = {r:.3f}, p = {p:.3f}")

    fig.suptitle(f"{mqi} vs Classroom Responsiveness", fontsize=14)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Print correlation table
# -----------------------------
print("\nClassroom-Level Correlations\n")
print("-" * 60)

results = []

for mqi in mqi_vars:
    for resp in resp_vars:

        df_clean = merged[[resp, mqi]].dropna()

        if len(df_clean) < 3:
            continue

        x = df_clean[resp]
        y = df_clean[mqi]

        r, p = pearsonr(x, y)

        results.append({
            "Metadata": mqi,
            "Responsiveness_Var": resp,
            "N": len(df_clean),
            "Pearson_r": r,
            "p_value": p
        })

results_df = pd.DataFrame(results)

# Print nicely formatted
for mqi in mqi_vars:
    print(f"\n=== {mqi} ===")
    subset = results_df[results_df["Metadata"] == mqi]

    for _, row in subset.iterrows():
        print(
            f"{row['Responsiveness_Var']:<22} "
            f"N={int(row['N']):<3}  "
            f"r={row['Pearson_r']:.3f}  "
            f"p={row['p_value']:.4f}"
        )

print("\n" + "-" * 60)