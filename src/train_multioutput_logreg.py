"""
Train a MultiOutputClassifier(LogisticRegression) on manual features,
evaluate on a separate test CSV, and save predictions.

Example:
  .venv/bin/python src/train_multioutput_logreg.py \
    --train_csv NCTE_Transcripts/processed/annotations/sampled_teacher_utt_pseudolabels-manual_features.csv \
    --test_csv NCTE_Transcripts/processed/annotations/agreed_annotations-manual_features.csv \
    --output_predictions_csv NCTE_Transcripts/processed/annotations/agreed_annotations-multioutput_predictions.csv
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
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


def _label_variants(base_name: str) -> List[str]:
    return [f"{base_name}_final", base_name]


def _resolve_label_cols(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for base in LABEL_BASE_TO_SHORT.keys():
        chosen = None
        for cand in _label_variants(base):
            if cand in df.columns:
                chosen = cand
                break
        if chosen is None:
            raise ValueError(f"Could not find label column for base name: {base}")
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


def _build_model() -> Pipeline:
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(base)),
        ]
    )


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_predictions_csv", type=str, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    id_col = "target_comb_idx" if "target_comb_idx" in test_df.columns else "row_index"
    if id_col == "row_index" and "row_index" not in test_df.columns:
        test_df[id_col] = np.arange(len(test_df))

    train_label_cols = _resolve_label_cols(train_df)
    test_label_cols = _resolve_label_cols(test_df)
    feat_cols = _feature_columns(test_df, id_col, test_label_cols)
    missing_in_train = [c for c in feat_cols if c not in train_df.columns]
    if missing_in_train:
        raise ValueError(f"Train CSV missing feature columns: {missing_in_train}")

    # Use rows with complete labels across all 4 tasks for train/test.
    y_train_df = pd.DataFrame(
        {
            short: _to_binary(train_df[train_label_cols[base]])
            for base, short in LABEL_BASE_TO_SHORT.items()
        }
    )
    y_test_df = pd.DataFrame(
        {
            short: _to_binary(test_df[test_label_cols[base]])
            for base, short in LABEL_BASE_TO_SHORT.items()
        }
    )

    train_mask = y_train_df.notna().all(axis=1)
    test_mask = y_test_df.notna().all(axis=1)

    X_train = train_df.loc[train_mask, feat_cols].to_numpy(dtype=np.float32)
    Y_train = y_train_df.loc[train_mask, :].to_numpy(dtype=np.int32)
    X_test_eval = test_df.loc[test_mask, feat_cols].to_numpy(dtype=np.float32)
    Y_test = y_test_df.loc[test_mask, :].to_numpy(dtype=np.int32)
    X_test_all = test_df[feat_cols].to_numpy(dtype=np.float32)

    model = _build_model()
    model.fit(X_train, Y_train)

    Y_pred_eval = model.predict(X_test_eval)
    clf = model.named_steps["clf"]
    prob_list = clf.predict_proba(model.named_steps["scaler"].transform(X_test_all))

    shorts = list(LABEL_BASE_TO_SHORT.values())
    metrics_rows = []
    for i, short in enumerate(shorts):
        acc, prec, rec, f1 = _evaluate(Y_test[:, i], Y_pred_eval[:, i])
        metrics_rows.append(
            {
                "label": short,
                "n_train": int(len(Y_train)),
                "n_test": int(len(Y_test)),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

    Y_pred_all = model.predict(X_test_all).astype(int)
    pred_df = pd.DataFrame({id_col: test_df[id_col]})
    for i, short in enumerate(shorts):
        pred_df[f"pred_{short}"] = Y_pred_all[:, i]
        pred_df[f"prob_{short}"] = prob_list[i][:, 1]
        pred_df[f"true_{short}"] = y_test_df[short]

    pred_df.to_csv(args.output_predictions_csv, index=False)
    print(f"Saved predictions CSV to {args.output_predictions_csv}")
    print(f"Rows: {len(pred_df)}")

    metrics_df = pd.DataFrame(metrics_rows)
    print("\nPer-label held-out metrics:")
    print(
        metrics_df[
            ["label", "n_train", "n_test", "accuracy", "precision", "recall", "f1"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()

