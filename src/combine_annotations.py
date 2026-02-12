import pandas as pd
import numpy as np
import itertools

# ---- SETTINGS ----
data_folder = "NCTE_Transcripts/processed/annotations/"
file_path = data_folder + "annotations.xlsx"
annotator_names = ["Jason", "Evelyn", "Elisabeth", "Mrs. Cousins"]

labels = [
    "R1: References prior student content",
    "R2: Builds on student content",
    "R3: Invites further student thinking",
    "C1. No student content available (N/A)",
    "C2. Want annotation review"
]

id_col = "OBSID"

# Context + metadata columns to preserve
meta_cols = [
    "target_comb_idx", "target_text",
    "ctx_-4_speaker", "ctx_-4_text",
    "ctx_-3_speaker", "ctx_-3_text",
    "ctx_-2_speaker", "ctx_-2_text",
    "ctx_-1_speaker", "ctx_-1_text",
    "ctx_0_speaker", "ctx_0_text",
    "ctx_1_speaker", "ctx_1_text",
    "ctx_2_speaker", "ctx_2_text"
]

# ---- LOAD AND CONCATENATE ----
dfs = []

for name in annotator_names:
    df = pd.read_excel(file_path, sheet_name=name)

    # Fill blank label cells with FALSE
    df[labels] = df[labels].fillna(False)

    # Convert TRUE/FALSE strings to booleans if needed
    df[labels] = df[labels].replace({"TRUE": True, "FALSE": False})

    df["annotator"] = name
    dfs.append(df)

all_annotations = pd.concat(dfs, ignore_index=True)

# ---- CREATE FINAL DATAFRAME BASE ----
# Keep one copy of metadata per OBSID
metadata = all_annotations.drop_duplicates(subset=id_col)[[id_col] + meta_cols]

final_df = metadata.copy()

# ---- PROCESS EACH LABEL ----
for label in labels:

    pivot = all_annotations.pivot_table(
        index=id_col,
        columns="annotator",
        values=label,
        aggfunc="first"
    )

    def resolve_row(row):
        values = row.dropna().values

        if len(values) == 0:
            return 0  # safety fallback

        if all(values == True):
            return 1
        elif all(values == False):
            return 0
        else:
            return "REVIEW"

    resolved = pivot.apply(resolve_row, axis=1)

    final_df = final_df.merge(
        resolved.rename(label + "_final"),
        left_on=id_col,
        right_index=True,
        how="left"
    )

# ---- SAVE OUTPUT ----
output_filepath = data_folder + "final_annotations_for_review.csv"
final_df.to_csv(output_filepath, index=False)

print(f"Done. File saved as {output_filepath}")