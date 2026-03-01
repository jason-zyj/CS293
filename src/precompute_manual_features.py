"""
Compute and cache heuristic text features from utterance + context CSV.

Example:
python src/precompute_manual_features.py \
  --input_csv NCTE_Transcripts/processed/annotations/synthetic_annotations.csv \
  --output_csv NCTE_Transcripts/processed/annotations/synthetic_manual_features.csv
"""

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from manual_features import FEATURE_NAMES, feature_matrix


LABEL_COLS: List[str] = [
    "R1: References prior student content",
    "R2: Builds on student content",
    "R3: Invites further student thinking",
    "C1. No student content available (N/A)",
    "R1: References prior student content_final",
    "R2: Builds on student content_final",
    "R3: Invites further student thinking_final",
    "C1. No student content available (N/A)_final",
]


def _to_binary_array(df: pd.DataFrame, cols: List[str]) -> Optional[np.ndarray]:
    if not all(c in df.columns for c in cols):
        return None

    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes"}:
                return 1.0
            if s in {"false", "0", "no"}:
                return 0.0
            if s == "review":
                return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    lab = df[cols].applymap(conv).to_numpy(dtype=np.float32)
    return lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_npz", type=str, default=None)
    parser.add_argument("--include_labels", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    label_cols_in_df = [c for c in LABEL_COLS if c in df.columns]

    X, names = feature_matrix(df)
    y = _to_binary_array(df, label_cols_in_df)

    if "target_comb_idx" in df.columns:
        ids = df["target_comb_idx"].to_numpy()
        id_col_name = "target_comb_idx"
    else:
        ids = np.arange(len(df))
        id_col_name = "row_index"

    feat_df = pd.DataFrame(X, columns=names)
    feat_df.insert(0, id_col_name, ids)

    if args.include_labels and y is not None:
        label_df = pd.DataFrame(y, columns=label_cols_in_df)
        feat_df = pd.concat([feat_df, label_df], axis=1)

    feat_df.to_csv(args.output_csv, index=False)
    print(f"Saved feature CSV to {args.output_csv}")
    print(f"rows: {len(feat_df)}")
    print(f"feature count: {len(FEATURE_NAMES)}")
    if args.include_labels:
        if y is not None:
            print("labels included in CSV.")
        else:
            print("labels requested, but not found in input CSV.")

    if args.output_npz:
        save_payload = {
            "features": X,
            "feature_names": np.array(names, dtype=object),
            "ids": ids,
        }
        if y is not None:
            save_payload["labels"] = y
        np.savez(args.output_npz, **save_payload)
        print(f"Saved optional NPZ cache to {args.output_npz}")


if __name__ == "__main__":
    main()
