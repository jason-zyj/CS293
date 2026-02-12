import pandas as pd
import itertools
from sklearn.metrics import cohen_kappa_score

# ---- SETTINGS ----
file_path = "NCTE_Transcripts/processed/annotations/annotations.xlsx"
annotator_names = ["Jason", "Evelyn", "Elisabeth", "Mrs. Cousins"]
labels = [
    "R1: References prior student content",
    "R2: Builds on student content",
    "R3: Invites further student thinking",
    "C1. No student content available (N/A)",
    "C2. Want annotation review"
]
id_col = "target_comb_idx"

# ---- LOAD ALL SHEETS ----
annotators = {}
for name in annotator_names:
    df = pd.read_excel(file_path, sheet_name=name)

    # Fill blank label cells with FALSE
    df[labels] = df[labels].fillna(False)

    # Convert TRUE/FALSE strings to actual booleans if needed
    df[labels] = df[labels].replace(
        {"TRUE": True, "FALSE": False}
    )

    annotators[name] = df

# ---- COMPUTE IRR FOR ALL PAIRS ----
results = []

for name1, name2 in itertools.combinations(annotator_names, 2):

    df1 = annotators[name1]
    df2 = annotators[name2]

    # Merge only overlapping OBSIDs
    merged = df1.merge(
        df2,
        on=id_col,
        suffixes=(f"_{name1}", f"_{name2}")
    )

    if len(merged) == 0:
        continue  # skip pairs with no overlap

    print(f"\n=== {name1} vs {name2} ===")
    print(f"Overlapping examples: {len(merged)}")

    for label in labels:
        col1 = f"{label}_{name1}"
        col2 = f"{label}_{name2}"

        agreement = (merged[col1] == merged[col2]).mean()
        kappa = cohen_kappa_score(merged[col1], merged[col2])

        print(f"\n{label}")
        print(f"Percent agreement: {agreement:.3f}")
        print(f"Cohen's kappa:    {kappa:.3f}")

        results.append({
            "Annotator 1": name1,
            "Annotator 2": name2,
            "Label": label,
            "N_overlap": len(merged),
            "Percent_agreement": agreement,
            "Cohen_kappa": kappa
        })

# ---- OPTIONAL: SAVE RESULTS ----
# results_df = pd.DataFrame(results)
# results_df.to_csv("irr_results.csv", index=False)