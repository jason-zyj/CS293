import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path

############################################
# 1. Load data (robust paths)
############################################

ROOT = Path(__file__).resolve().parents[1]  # .../CS293 (repo root)
DATA_DIR = ROOT / "NCTE_Transcripts/processed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # creates outputs/ if missing


DATA_PATH = DATA_DIR / "ncte_single_utterances_cleaned.csv"
META_PATH = DATA_DIR / "transcript_metadata.csv"

print("Script root:", ROOT)
print("DATA_DIR:", DATA_DIR)
print("DATA_PATH:", DATA_PATH)
print("DATA exists?", DATA_PATH.exists())
print("META_PATH:", META_PATH)
print("META exists?", META_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Utterance file not found: {DATA_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {META_PATH}")




df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace", low_memory=False)
meta = pd.read_csv(META_PATH, encoding="utf-8", encoding_errors="replace", low_memory=False)

print("Initial utterance dataset shape:", df.shape)
print("Utterance columns:", df.columns.tolist())
print("Initial metadata dataset shape:", meta.shape)
print("Metadata columns:", meta.columns.tolist())

############################################
# 2. Ensure join key matches
############################################

df["OBSID"] = df["OBSID"].astype(str)
meta["OBSID"] = meta["OBSID"].astype(str)

############################################
# 3. Clean outcome variable (STATEVA_M)
############################################

if "STATEVA_M" not in meta.columns:
    raise ValueError("STATEVA_M not found in transcript_metadata.csv columns.")

meta["STATEVA_M_clean"] = pd.to_numeric(meta["STATEVA_M"], errors="coerce")
# Trim extreme outliers (adjust threshold if needed)
meta.loc[meta["STATEVA_M_clean"].abs() > 5, "STATEVA_M_clean"] = np.nan

############################################
# 4. Sanity check required columns in utterance df
############################################

required_cols = [
    "OBSID",
    "teacher_after_student",
    "turn_idx",
    "teacher_acknowledgment",
    "teacher_reformulation_overlap",
    "num_words_v2",
    "is_student",
    "text",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in utterance file: {missing}")

############################################
# 5. Core feature engineering: responsiveness after student turns
############################################

tas = df[df["teacher_after_student"] == 1].copy()

# Ensure consistent order within each lesson
tas = tas.sort_values(["OBSID", "turn_idx"])

# Position of each teacher-after-student turn within lesson (0, 1, 2, ...)
tas["pos"] = tas.groupby("OBSID").cumcount()

# Number of teacher-after-student turns per lesson
tas["n_in_lesson"] = tas.groupby("OBSID")["pos"].transform("max") + 1

# Assign early/mid/late by thirds (terciles)
# If a lesson has fewer than 3 such turns, we set segment to NaN.
tas["segment"] = np.where(
    tas["n_in_lesson"] < 3,
    np.nan,
    np.where(
        tas["pos"] < tas["n_in_lesson"] / 3,
        "early",
        np.where(tas["pos"] < 2 * tas["n_in_lesson"] / 3, "mid", "late"),
    ),
)

# Aggregate per OBSID x segment
seg = (tas.dropna(subset=["segment"])
         .groupby(["OBSID", "segment"])
         .agg(
             n_t_after_s=("teacher_after_student", "size"),
             ack_rate=("teacher_acknowledgment", "mean"),
             reform_overlap_rate=("teacher_reformulation_overlap", "mean"),
             avg_words=("num_words_v2", "mean"),
         )
         .reset_index())

# Pivot to wide: early/mid/late columns
wide = seg.pivot(index="OBSID", columns="segment",
                 values=["ack_rate", "reform_overlap_rate", "n_t_after_s"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()

# Ensure expected columns exist (prevents KeyErrors when some segments are absent)
for c in ["ack_rate_early", "ack_rate_late", "reform_overlap_rate_early", "reform_overlap_rate_late"]:
    if c not in wide.columns:
        wide[c] = np.nan

# Compute early→late change
wide["ack_change_late_minus_early"] = wide["ack_rate_late"] - wide["ack_rate_early"]
wide["reform_change_late_minus_early"] = wide["reform_overlap_rate_late"] - wide["reform_overlap_rate_early"]

############################################
# 6. Baseline lesson-level features from all utterances
############################################

all_lesson = (df.groupby("OBSID")
                .agg(
                    n_turns=("text", "size"),
                    pct_student_turns=("is_student", "mean"),
                    total_words=("num_words_v2", "sum"),
                )
                .reset_index())

############################################
# 7. Merge features + outcome
############################################

analysis = (meta[["OBSID", "STATEVA_M_clean"]]
            .merge(all_lesson, on="OBSID", how="left")
            .merge(wide, on="OBSID", how="left"))

############################################
# 8. Deliverable 1: descriptive distributions
############################################

cols_to_describe = [
    "ack_rate_early", "ack_rate_late",
    "reform_overlap_rate_early", "reform_overlap_rate_late"
]
existing = [c for c in cols_to_describe if c in analysis.columns]

print("\nDescriptive stats (responsiveness features):")
print(analysis[existing].describe())

############################################
# 9. Deliverable 2: within-lesson change story
############################################

if "reform_change_late_minus_early" in analysis.columns:
    pct_increase = np.mean(analysis["reform_change_late_minus_early"].dropna() > 0)
    print("\nPct lessons with increased reformulation overlap late vs early:", pct_increase)
else:
    print("\nCannot compute reform_change_late_minus_early (missing features).")

############################################
# 10. Deliverable 3: correlations with STATEVA_M outcome
############################################

OUTCOME = "STATEVA_M_clean"

def corr(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 50 else np.nan

y = analysis[OUTCOME]
print(f"\nOutcome variable: {OUTCOME}")
print(f"N non-missing outcome: {y.notna().sum()} out of {len(analysis)} transcripts\n")

predictors = {
    "ack_rate_late": analysis.get("ack_rate_late"),
    "reform_overlap_rate_late": analysis.get("reform_overlap_rate_late"),
    "ack_change_late_minus_early": analysis.get("ack_change_late_minus_early"),
    "reform_change_late_minus_early": analysis.get("reform_change_late_minus_early"),
    # Optional baselines
    "pct_student_turns": analysis.get("pct_student_turns"),
    "total_words": analysis.get("total_words"),
    "n_turns": analysis.get("n_turns"),
}

for name, x in predictors.items():
    if x is None:
        continue
    r = corr(x, y)
    n = (x.notna() & y.notna()).sum()
    print(f"Corr({name}, {OUTCOME}) = {r:.3f}  (N={n})")

############################################
# Optional: save the merged analysis table for later plotting
############################################
# OUT_CSV = ROOT / "src" / "responsiveness_eda_output.csv"
# analysis.to_csv(OUT_CSV, index=False)
# print(f"\nSaved merged analysis table to: {OUT_CSV}")

desc = analysis[existing].describe()
desc.to_csv(ROOT / "outputs" / "responsiveness_describe.csv", index=True)
print("Saved:", ROOT / "outputs" / "responsiveness_describe.csv")