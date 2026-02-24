import pandas as pd
import numpy as np

# Paths
csv_path = "NCTE_Transcripts/processed/annotations/final_annotations_for_review.csv"
orig_xlsx_path = "NCTE_Transcripts/processed/annotations/annotations.xlsx"
output_xlsx_path = "NCTE_Transcripts/processed/annotations/third_annotations.xlsx"

# Annotators in order
annotators = ["Evelyn", "Jason", "Elisabeth", "Mrs. Cousins"]

# Load final review CSV
df = pd.read_csv(csv_path)

# Columns to check for REVIEW
review_cols = ['R1: References prior student content_final',
               'R2: Builds on student content_final',
               'R3: Invites further student thinking_final',
               'C1. No student content available (N/A)_final']

# Filter rows with REVIEW in any of these columns
df_review = df[df[review_cols].isin(['REVIEW']).any(axis=1)].copy()

# Load original annotations to check who annotated what
orig_xlsx = pd.ExcelFile(orig_xlsx_path)
annotations_per_annotator = {name: pd.read_excel(orig_xlsx, sheet_name=name) for name in annotators}

# Create dict for new DataFrames for each annotator tab
annotator_dfs = {name: [] for name in annotators}

# Helper function to decide third annotator
def choose_third_annotator(existing):
    existing = set(existing)
    if existing == {"Jason", "Evelyn"}:
        return "Elisabeth"
    elif existing == {"Evelyn", "Elisabeth"}:
        return "Jason"
    elif existing == {"Elisabeth", "Mrs. Cousins"}:
        return np.random.choice(["Evelyn", "Jason"])
    elif existing == {"Mrs. Cousins", "Jason"}:
        return "Elisabeth"
    else:
        # Fallback if only one annotator or something else
        remaining = [a for a in annotators if a not in existing]
        return remaining[0] if remaining else None

# Assign each REVIEW row
for _, row in df_review.iterrows():
    target_id = row['target_comb_idx']
    
    # Determine which annotators already annotated this example
    existing_annotators = []
    for name in annotators:
        if target_id in annotations_per_annotator[name]['target_comb_idx'].values:
            existing_annotators.append(name)
    
    # Choose the third annotator
    third = choose_third_annotator(existing_annotators)
    
    if third:
        annotator_dfs[third].append(row)

# Convert lists to DataFrames and preserve column order
for name in annotators:
    if annotator_dfs[name]:
        annotator_dfs[name] = pd.DataFrame(annotator_dfs[name])[df.columns]
    else:
        annotator_dfs[name] = pd.DataFrame(columns=df.columns)

# Write to new Excel with the same tabs
with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
    for name in annotators:
        annotator_dfs[name].to_excel(writer, sheet_name=name, index=False)

print(f"Review annotations saved to {output_xlsx_path}")