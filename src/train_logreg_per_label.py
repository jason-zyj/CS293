"""
Train separate logistic regression models for each label variable (R1/R2/R3/C1)
from manual-feature CSVs, save predictions to CSV, and report metrics when
ground-truth labels are available in the test CSV.

Example:
  .venv/bin/python src/train_logreg_per_label.py \
    --train_csv NCTE_Transcripts/processed/annotations/sampled_teacher_utt_pseudolabels-manual_features.csv \
    --test_csv NCTE_Transcripts/processed/annotations/agreed_annotations-manual_features.csv \
    --output_predictions_csv NCTE_Transcripts/processed/annotations/agreed_annotations-logreg_predictions.csv
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_BASE_TO_SHORT: Dict[str, str] = {
    "R1: References prior student content": "R1",
    "R2: Builds on student content": "R2",
    "R3: Invites further student thinking": "R3",
    "C1. No student content available (N/A)": "C1",
}


def _to_binary(series: pd.Series) -> pd.Series:
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

    return series.apply(conv)


def _build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def _label_variants(base_name: str) -> List[str]:
    return [f"{base_name}_final", base_name]


def _resolve_label_cols(df: pd.DataFrame, required: bool = True) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for base in LABEL_BASE_TO_SHORT.keys():
        chosen = None
        for cand in _label_variants(base):
            if cand in df.columns:
                chosen = cand
                break
        if chosen is None and required:
            raise ValueError(f"Could not find label column for base name: {base}")
        if chosen is not None:
            resolved[base] = chosen
    return resolved


def _feature_columns(df: pd.DataFrame, id_col: str, resolved_label_cols: Dict[str, str]) -> List[str]:
    exclude = set(resolved_label_cols.values())
    exclude.add(id_col)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_predictions_csv", type=str, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    id_col = "target_comb_idx" if "target_comb_idx" in test_df.columns else None
    if id_col is None:
        id_col = "row_index"
        test_df[id_col] = np.arange(len(test_df))

    train_label_cols = _resolve_label_cols(train_df, required=True)
    test_label_cols = _resolve_label_cols(test_df, required=False)

    feat_cols = _feature_columns(test_df, id_col, test_label_cols)
    if not feat_cols:
        raise ValueError("No numeric feature columns found.")
    missing_in_train = [c for c in feat_cols if c not in train_df.columns]
    if missing_in_train:
        raise ValueError(f"Train CSV missing feature columns: {missing_in_train}")

    X_train_all = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_test_all = test_df[feat_cols].to_numpy(dtype=np.float32)
    pred_df = pd.DataFrame({id_col: test_df[id_col]})
    metrics_rows = []

    for label_base, short in LABEL_BASE_TO_SHORT.items():
        y_train_series = _to_binary(train_df[train_label_cols[label_base]])

        train_valid_mask = y_train_series.notna()

        X_train = X_train_all[train_valid_mask.values]
        y_train = y_train_series[train_valid_mask].to_numpy(dtype=np.int32)

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{short}: only one class present after filtering; cannot train logistic regression.")

        model = _build_model()
        model.fit(X_train, y_train)

        pred_all = model.predict(X_test_all).astype(int)
        prob_all = model.predict_proba(X_test_all)[:, 1]

        pred_df[f"pred_{short}"] = pred_all
        pred_df[f"prob_{short}"] = prob_all
        if label_base in test_label_cols:
            y_test_series = _to_binary(test_df[test_label_cols[label_base]])
            pred_df[f"true_{short}"] = y_test_series

            test_valid_mask = y_test_series.notna()
            X_test_eval = X_test_all[test_valid_mask.values]
            y_test = y_test_series[test_valid_mask].to_numpy(dtype=np.int32)
            if len(y_test) > 0:
                y_test_pred = model.predict(X_test_eval)
                acc, prec, rec, f1 = _evaluate(y_test, y_test_pred)
                metrics_rows.append(
                    {
                        "label": short,
                        "n_train_total": int(len(y_train)),
                        "n_test_total": int(len(y_test)),
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                    }
                )

    pred_df.to_csv(args.output_predictions_csv, index=False)
    print(f"Saved predictions CSV to {args.output_predictions_csv}")
    print(f"Rows: {len(pred_df)}")

    metrics_df = pd.DataFrame(metrics_rows)
    if len(metrics_df) > 0:
        print("\nPer-label held-out metrics:")
        print(
            metrics_df[
                ["label", "n_train_total", "n_test_total", "n_train", "n_test", "accuracy", "precision", "recall", "f1"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    else:
        print("\nNo held-out metrics computed: test CSV does not contain usable ground-truth labels.")


if __name__ == "__main__":
    main()
