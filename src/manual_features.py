import re
import string
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS = set(ENGLISH_STOP_WORDS)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

RE_BRACKET = re.compile(r"\[.*?\]")
RE_NUMBER = re.compile(r"\b\d+(?:[./]\d+)?\b")
RE_WORD = re.compile(r"[a-z0-9]+")


MATH_VOCAB = {
    "add", "added", "adding", "sum", "plus", "minus", "subtract", "difference",
    "multiply", "times", "product", "divide", "divided", "quotient", "equal",
    "equals", "equation", "expression", "fraction", "fractions", "decimal",
    "decimals", "percent", "ratio", "proportion", "numerator", "denominator",
    "whole", "part", "parts", "value", "solve", "solution", "strategy", "method",
    "model", "area", "perimeter", "length", "width", "number", "numbers",
}

REASONING_MARKERS = {
    "because", "since", "therefore", "so", "if", "then", "means", "reason",
    "why", "prove", "explain", "justify", "shows",
}

EVAL_ONLY_TOKENS = {
    "yes", "yeah", "yep", "right", "correct", "good", "great", "nice", "no", "nope",
}

MANAGEMENT_MARKERS = {
    "take out", "turn to", "sit down", "line up", "be quiet", "quiet", "listen up",
    "eyes on", "pencils down", "stop talking", "talking", "desk", "desks", "page",
    "pages", "homework", "worksheet", "materials", "seat", "seats",
}

FOLLOWUP_PROMPTS = {
    "say more", "can you explain", "why do you", "how do you know", "what makes",
    "do you agree", "who can add", "another way", "different way", "build on",
    "can someone", "who wants to", "convince us", "prove it",
}

REFERENCE_PATTERNS = (
    r"\byou said\b",
    r"\byour idea\b",
    r"\bthat means\b",
    r"\bthis means\b",
    r"\bso you\b",
    r"\bas you said\b",
)

OPEN_Q_STARTERS = {
    "why", "how", "what", "which", "could", "would", "can", "explain", "justify",
}
YESNO_STARTERS = {
    "is", "are", "do", "does", "did", "can", "could", "will", "would", "have", "has",
}


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def normalize_text(text: str) -> str:
    text = _safe_text(text).lower().strip()
    text = RE_BRACKET.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def raw_tokens(text: str) -> List[str]:
    return RE_WORD.findall(normalize_text(text))


def content_tokens(text: str) -> List[str]:
    return [t for t in raw_tokens(text) if t not in STOP_WORDS]


def number_tokens(text: str) -> List[str]:
    return RE_NUMBER.findall(_safe_text(text).lower())


def _ratio_overlap(a: Set[str], b: Set[str], denom: int) -> float:
    if denom <= 0:
        return 0.0
    return len(a.intersection(b)) / float(denom)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / float(len(union))


def _contains_any_phrase(text: str, phrases: Iterable[str]) -> int:
    return int(any(p in text for p in phrases))


def _contains_any_regex(text: str, patterns: Sequence[str]) -> int:
    return int(any(re.search(p, text) for p in patterns))


def _contains_ngram_overlap(tokens_a: List[str], tokens_b: List[str], n: int) -> int:
    if len(tokens_a) < n or len(tokens_b) < n:
        return 0
    ngrams_a = {" ".join(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)}
    ngrams_b = {" ".join(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)}
    return int(bool(ngrams_a.intersection(ngrams_b)))


def _is_open_question(text: str) -> int:
    if "?" not in text:
        return 0
    toks = raw_tokens(text)
    if not toks:
        return 0
    first = toks[0]
    if first in OPEN_Q_STARTERS and first not in YESNO_STARTERS:
        return 1
    if any(k in text for k in ("why", "how", "explain", "justify", "what makes")):
        return 1
    return 0


def _is_imperative(text: str) -> int:
    toks = raw_tokens(text)
    if not toks:
        return 0
    return int(toks[0] in {"look", "tell", "write", "turn", "show", "take", "listen", "stop"})


def _student_content_available(row: pd.Series) -> int:
    for i in [-1, -2, -3, -4]:
        speaker = _safe_text(row.get(f"ctx_{i}_speaker", "")).lower()
        text = _safe_text(row.get(f"ctx_{i}_text", ""))
        if "student" in speaker:
            if len(content_tokens(text)) >= 2 and "[inaudible]" not in text.lower():
                return 1
    return 0


FEATURE_NAMES = [
    "overlap_ctxm1_content_ratio",
    "overlap_ctxm1_jaccard",
    "number_overlap_ratio",
    "math_vocab_overlap_ratio",
    "echo_ngram_2_flag",
    "echo_ngram_3_flag",
    "reference_phrase_flag",
    "teacher_eval_only_flag",
    "why_how_prompt_flag",
    "question_mark_flag",
    "open_question_flag",
    "followup_prompt_flag",
    "reasoning_marker_count",
    "revoice_plus_extension_flag",
    "correction_with_reason_flag",
    "management_language_flag",
    "inaudible_ctxm1_flag",
    "student_content_available_flag",
    "target_length_tokens",
    "imperative_verb_flag",
    "student_to_teacher_turn_flag",
]


def compute_features_for_row(row: pd.Series) -> Dict[str, float]:
    target_raw = _safe_text(row.get("target_text", ""))
    ctxm1_raw = _safe_text(row.get("ctx_-1_text", ""))
    target = normalize_text(target_raw)
    ctxm1 = normalize_text(ctxm1_raw)

    t_raw = raw_tokens(target)
    c_raw = raw_tokens(ctxm1)
    t_content = set(content_tokens(target))
    c_content = set(content_tokens(ctxm1))

    t_nums = set(number_tokens(target_raw))
    c_nums = set(number_tokens(ctxm1_raw))
    t_math = {t for t in t_content if t in MATH_VOCAB}
    c_math = {t for t in c_content if t in MATH_VOCAB}

    overlap_ratio = _ratio_overlap(t_content, c_content, len(t_content))
    jaccard = _jaccard(t_content, c_content)
    num_overlap = _ratio_overlap(t_nums, c_nums, len(t_nums))
    math_overlap = _ratio_overlap(t_math, c_math, len(t_math))

    ref_flag = _contains_any_regex(target, REFERENCE_PATTERNS)
    question_flag = int("?" in target_raw)
    open_q_flag = _is_open_question(target_raw.lower())
    followup_flag = _contains_any_phrase(target, FOLLOWUP_PROMPTS)
    management_flag = _contains_any_phrase(target, MANAGEMENT_MARKERS)
    imperative_flag = _is_imperative(target_raw)

    reasoning_count = sum(1 for tok in t_raw if tok in REASONING_MARKERS)
    why_how_flag = int(any(x in target for x in ("why", "how", "explain", "justify", "what makes")))
    eval_only_flag = int(set(t_raw).issubset(EVAL_ONLY_TOKENS) and len(t_raw) <= 5 and overlap_ratio < 0.2)

    correction_with_reason = int(
        ("not" in t_raw or "actually" in t_raw or "but" in t_raw)
        and reasoning_count > 0
    )
    revoice_plus_extension = int(overlap_ratio > 0.2 and reasoning_count > 0)

    inaudible_ctx = int("[inaudible]" in ctxm1_raw.lower() or len(c_raw) == 0)
    student_available = _student_content_available(row)
    speaker_prev = _safe_text(row.get("ctx_-1_speaker", "")).lower()
    student_to_teacher = int("student" in speaker_prev)

    return {
        "overlap_ctxm1_content_ratio": overlap_ratio,
        "overlap_ctxm1_jaccard": jaccard,
        "number_overlap_ratio": num_overlap,
        "math_vocab_overlap_ratio": math_overlap,
        "echo_ngram_2_flag": _contains_ngram_overlap(t_raw, c_raw, 2),
        "echo_ngram_3_flag": _contains_ngram_overlap(t_raw, c_raw, 3),
        "reference_phrase_flag": ref_flag,
        "teacher_eval_only_flag": eval_only_flag,
        "why_how_prompt_flag": why_how_flag,
        "question_mark_flag": question_flag,
        "open_question_flag": open_q_flag,
        "followup_prompt_flag": followup_flag,
        "reasoning_marker_count": float(reasoning_count),
        "revoice_plus_extension_flag": revoice_plus_extension,
        "correction_with_reason_flag": correction_with_reason,
        "management_language_flag": management_flag,
        "inaudible_ctxm1_flag": inaudible_ctx,
        "student_content_available_flag": student_available,
        "target_length_tokens": float(len(t_raw)),
        "imperative_verb_flag": imperative_flag,
        "student_to_teacher_turn_flag": student_to_teacher,
    }


def feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    rows = [compute_features_for_row(row) for _, row in df.iterrows()]
    feat_df = pd.DataFrame(rows)
    feat_df = feat_df[FEATURE_NAMES]
    return feat_df.to_numpy(dtype=np.float32), FEATURE_NAMES
