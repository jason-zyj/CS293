#!/usr/bin/env python3
"""Generate pseudo-labels (R1/R2/R3/C1) for teacher utterances using gpt-5-nano.

Usage:
  export OPENAI_API_KEY=...
  python src/generate_pseudolabels.py \
    --input NCTE_Transcripts/processed/annotations/sampled_teacher_utt_pseudolabels.csv \
    --output NCTE_Transcripts/processed/annotations/sampled_teacher_utt_pseudolabels_labeled.csv

Notes:
- Calls OpenAI Chat Completions API with JSON schema output.
- Supports resume: rows with existing output labels are skipped.
- Saves progress every N rows (default: 20).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from typing import Dict, List
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an expert educational discourse annotator. "
    "Label the TARGET teacher utterance using only provided context. "
    "Be conservative for R1: only mark R1=1 when there is explicit evidence of uptake of prior student content. "
    "Return strictly valid JSON matching the schema."
)

R2_REASONING_MARKERS = (
    "because",
    "that means",
    "if ",
    "then ",
    "equals",
    "equal",
    "divided",
    "times",
    "plus",
    "minus",
    "fraction",
    "numerator",
    "denominator",
)


def normalize_binary(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "yes", "true", "y"}:
            return 1
    return 0


def build_user_prompt(row: Dict[str, str]) -> str:
    return f"""
Annotate the TARGET teacher utterance with binary labels.

Definitions:
- R1 (References prior student content): teacher explicitly references student's immediately prior idea/answer/strategy.
- R2 (Builds on student content): teacher explains/modifies/extends student idea (teacher does cognitive work).
- R3 (Invites further student thinking): teacher prompts students to explain/justify/extend/revise (student does cognitive work).
- C1 (No student content available / N-A): utterance cannot be responsive (e.g., logistics/opening mgmt/inaudible prior student content).

Rules:
- Use only transcript context below.
- If there is no usable prior student content, R1=0 and R2=0.
- R2 can only be 1 if R1 is 1.
- R1 strict rule: set R1=1 only if TARGET explicitly repeats, paraphrases, or uses a concrete prior student idea/answer/strategy from recent context.
- R1=0 if TARGET is mainly classroom management/procedure, generic calling on students, generic prompting without uptake, or praise/affect without student-content uptake.
- R1=1 can still apply without naming a student if TARGET clearly restates a specific student-provided value/term/claim from recent context.
- R1=0 for generic question-only/callout turns that do not clearly restate a specific student-provided value/term/claim.
- R1=1 for short direct lexical echoes (including brief clarifications) when TARGET clearly repeats a student-provided value/term/claim from recent context.
- Words like "that/this/it/you" do NOT count as R1 evidence by themselves; they must clearly point to a specific prior student idea/answer/strategy.
- Before setting R1=1, identify the exact prior student span and overlapping content used in TARGET; if no concrete overlap, set R1=0.
- R2 strict rule: set R2=1 only when the teacher adds substantive cognitive work on the referenced student content (explains why, corrects with reasoning, extends/generalizes, connects representations, or models next reasoning steps).
- R2=0 for mere repetition, simple evaluation (e.g., "right/wrong/good"), procedural directions, or asking for another answer without adding reasoning.
- Do not set R2=1 unless TARGET contains teacher reasoning beyond quoting/restating student content.
- Tie-breaker: if uncertain, prefer R1=0.
- R1 decision examples:
  - Example (R1=0): student says "Rectangle." teacher says "What else could we have had?" -> generic solicitation, no uptake of the student's specific content.
  - Example (R1=0): student asks a procedural question and teacher says "Ask me that question one more time." -> interaction management, no uptake.
  - Example (R1=1): student says "One-fourth divided by what?" teacher says "It's on the board, one-fourth divided by 3." -> explicit restatement/clarification of student-raised content.
  - Example (R1=1): student gives a number/term and teacher immediately repeats that exact number/term to continue reasoning.
- Few-shot calibration:
  - Example A:
    ctx_-1: (student) Rectangle.
    TARGET: You could have had a rectangle. What else could we have had?
    output: {{"R1: References prior student content": 0, "R2: Builds on student content": 0, "R3: Invites further student thinking": 1, "C1. No student content available (N/A)": 0}}
  - Example B:
    ctx_-1: (student) One-fourth divided by what?
    TARGET: It's on the board, one-fourth divided by 3.
    output: {{"R1: References prior student content": 1, "R2: Builds on student content": 0, "R3: Invites further student thinking": 0, "C1. No student content available (N/A)": 0}}
  - Example C:
    ctx_-1: (student) 15
    TARGET: Put it here. Now what is 15 plus 15?
    output: {{"R1: References prior student content": 1, "R2: Builds on student content": 1, "R3: Invites further student thinking": 1, "C1. No student content available (N/A)": 0}}
- Output only JSON fields:  R1: References prior student content    R2: Builds on student content   R3: Invites further student thinking    C1: No student content available (N/A)
.

Context:
- target_comb_idx: {row.get('target_comb_idx', '')}
- TARGET: {row.get('target_text', '')}
- ctx_-4: ({row.get('ctx_-4_speaker', '')}) {row.get('ctx_-4_text', '')}
- ctx_-3: ({row.get('ctx_-3_speaker', '')}) {row.get('ctx_-3_text', '')}
- ctx_-2: ({row.get('ctx_-2_speaker', '')}) {row.get('ctx_-2_text', '')}
- ctx_-1: ({row.get('ctx_-1_speaker', '')}) {row.get('ctx_-1_text', '')}
- ctx_+1: ({row.get('ctx_1_speaker', '')}) {row.get('ctx_1_text', '')}
- ctx_+2: ({row.get('ctx_2_speaker', '')}) {row.get('ctx_2_text', '')}
""".strip()


def apply_label_post_rules(row: Dict[str, str], labels: Dict[str, int]) -> Dict[str, int]:
    out = dict(labels)

    # R2 recovery: if R1 is present and TARGET clearly carries reasoning/math-work language,
    # treat that as teacher cognitive work.
    target_lower = (row.get("target_text", "") or "").lower()
    if (
        out.get("R1: References prior student content", 0) == 1
        and out.get("R2: Builds on student content", 0) == 0
        and any(m in target_lower for m in R2_REASONING_MARKERS)
    ):
        out["R2: Builds on student content"] = 1

    # Keep rubric constraint.
    if out.get("R2: Builds on student content", 0) == 1 and out.get("R1: References prior student content", 0) == 0:
        out["R2: Builds on student content"] = 0

    return out


def call_openai_json(
    *,
    client: OpenAI,
    model: str,
    user_prompt: str,
    timeout_s: int,
    max_retries: int,
    retry_sleep_s: float,
) -> Dict[str, int]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "R1: References prior student content": {"type": "integer", "enum": [0, 1]},
            "R2: Builds on student content": {"type": "integer", "enum": [0, 1]},
            "R3: Invites further student thinking": {"type": "integer", "enum": [0, 1]},
            "C1. No student content available (N/A)": {"type": "integer", "enum": [0, 1]},
        },
        "required": [
            "R1: References prior student content",
            "R2: Builds on student content",
            "R3: Invites further student thinking",
            "C1. No student content available (N/A)",
        ],
    }

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "pseudo_labels",
                        "strict": True,
                        "schema": schema,
                    }
                },
                timeout=timeout_s,
            )
            content = resp.output_text
            label_obj = json.loads(content)

            out = {
                "R1: References prior student content": normalize_binary(label_obj.get("R1: References prior student content", 0)),
                "R2: Builds on student content": normalize_binary(label_obj.get("R2: Builds on student content", 0)),
                "R3: Invites further student thinking": normalize_binary(label_obj.get("R3: Invites further student thinking", 0)),
                "C1. No student content available (N/A)": normalize_binary(label_obj.get("C1. No student content available (N/A)", 0)),
            }

            # Hard constraint from rubric.
            if out["R2: Builds on student content"] == 1 and out["R1: References prior student content"] == 0:
                out["R2: Builds on student content"] = 0

            return out

        except RateLimitError as e:
            err_code = None
            if isinstance(getattr(e, "body", None), dict):
                err_code = e.body.get("code")
            if err_code == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI quota exceeded (insufficient_quota). "
                    "Check billing/credits at https://platform.openai.com/settings/organization/billing/overview"
                ) from e
            last_err = e
            retriable = True
            if not retriable or attempt == max_retries:
                break
            time.sleep(retry_sleep_s * attempt)
        except APIStatusError as e:
            err_code = None
            if isinstance(getattr(e, "body", None), dict):
                err_code = e.body.get("code")
            if err_code == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI quota exceeded (insufficient_quota). "
                    "Check billing/credits at https://platform.openai.com/settings/organization/billing/overview"
                ) from e
            last_err = RuntimeError(f"HTTP {e.status_code}: {str(e)[:600]}")
            retriable = e.status_code in {408, 409, 429, 500, 502, 503, 504}
            if not retriable or attempt == max_retries:
                break
            time.sleep(retry_sleep_s * attempt)
        except (APIConnectionError, APITimeoutError) as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_s * attempt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_s * attempt)


    raise RuntimeError(f"OpenAI request failed after {max_retries} tries: {last_err}")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pseudo-label teacher utterances with OpenAI model")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model")
    parser.add_argument("--start", type=int, default=0, help="Start row index (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many rows (0 = all)")
    parser.add_argument("--save-every", type=int, default=20, help="Save every N processed rows")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent API requests")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per row")
    parser.add_argument("--retry-sleep", type=float, default=0.8, help="Base sleep seconds for retries")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key)
    print(f"Using OpenAI key: {api_key[:4]}***{api_key[-4:]}")
    rows = read_csv_rows(args.input)
    if not rows:
        print("ERROR: Input CSV has no rows.", file=sys.stderr)
        return 1

    # Merge with existing output if present (resume behavior).
    existing_map: Dict[str, Dict[str, str]] = {}
    if os.path.exists(args.output):
        out_rows = read_csv_rows(args.output)
        for r in out_rows:
            k = r.get("target_comb_idx", "")
            if k:
                existing_map[k] = r

    base_fields = list(rows[0].keys())
    label_fields = ["R1: References prior student content", "R2: Builds on student content", "R3: Invites further student thinking", "C1. No student content available (N/A)"]
    fieldnames = base_fields[:]
    for lf in label_fields:
        if lf not in fieldnames:
            fieldnames.append(lf)

    # Prime rows with existing output labels where available.
    for r in rows:
        k = r.get("target_comb_idx", "")
        prior = existing_map.get(k)
        if prior:
            for lf in label_fields:
                if prior.get(lf, "") != "":
                    r[lf] = prior[lf]

    total = len(rows)
    start = max(args.start, 0)
    end = total if args.limit <= 0 else min(total, start + args.limit)

    processed = 0
    labeled = 0
    skipped = 0

    def label_one(index: int) -> tuple[int, Dict[str, int]]:
        row = rows[index]
        prompt = build_user_prompt(row)
        labels = call_openai_json(
            client=client,
            model=args.model,
            user_prompt=prompt,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep,
        )
        labels = apply_label_post_rules(row, labels)
        return index, labels

    to_process: List[int] = []
    for idx in range(start, end):
        row = rows[idx]
        already = all(str(row.get(c, "")).strip() in {"0", "1"} for c in label_fields)
        if already:
            skipped += 1
            continue
        to_process.append(idx)

    if args.concurrency <= 1:
        for idx in tqdm(to_process):
            _, labels = label_one(idx)
            for k, v in labels.items():
                rows[idx][k] = str(v)
            processed += 1
            labeled += 1
            if processed % args.save_every == 0:
                write_csv_rows(args.output, rows, fieldnames)
                print(f"Saved progress after {processed} processed rows (idx={idx}).")
    else:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                futures = [ex.submit(label_one, idx) for idx in to_process]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    idx, labels = fut.result()
                    for k, v in labels.items():
                        rows[idx][k] = str(v)
                    processed += 1
                    labeled += 1
                    if processed % args.save_every == 0:
                        write_csv_rows(args.output, rows, fieldnames)
                        print(f"Saved progress after {processed} processed rows (idx={idx}).")
        except Exception:
            write_csv_rows(args.output, rows, fieldnames)
            raise

    write_csv_rows(args.output, rows, fieldnames)

    print("Done.")
    print(f"Input rows: {total}")
    print(f"Range processed: [{start}, {end})")
    print(f"Newly labeled rows: {labeled}")
    print(f"Skipped already-labeled rows: {skipped}")
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
