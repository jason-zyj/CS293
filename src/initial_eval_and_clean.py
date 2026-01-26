import pandas as pd
import numpy as np
import re
from collections import Counter

############################################
# 1. Load data
############################################

DATA_PATH = "NCTE_Transcripts/ncte_single_utterances.csv"
df = pd.read_csv(DATA_PATH)

print("Initial dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())


############################################
# 2. Missingness analysis
############################################

print("\n===== MISSINGNESS ANALYSIS =====")
missingness = df.isna().mean().sort_values(ascending=False)
print(missingness)

# Count rows with missing critical fields
critical_cols = ["speaker", "text", "cleaned_text", "OBSID", "turn_idx"]
missing_critical = df[critical_cols].isna().any(axis=1).mean()
print(f"\nFraction of rows missing ≥1 critical field: {missing_critical:.3f}")


############################################
# 3. Noise & transcription artifacts
############################################

print("\n===== TRANSCRIPTION ARTIFACT ANALYSIS =====")

ARTIFACT_PATTERNS = {
    "inaudible": r"\b(?:inaudible|unintelligible)\b",
    "bracketed": r"\[.*?\]",
    "filled_pause": r"\b(?:um|uh|erm|mm)\b",
    "non_lexical": r"\b(?:hmm+|uhh+)\b"
}

artifact_stats = {}

for name, pattern in ARTIFACT_PATTERNS.items():
    artifact_stats[name] = df["cleaned_text"].str.contains(
        pattern, flags=re.IGNORECASE, regex=True, na=False
    ).mean()

artifact_df = pd.Series(artifact_stats).sort_values(ascending=False)
print(artifact_df)

# Very short utterances (e.g., "yes", "yeah")
short_utterances = (df["num_words"] <= 2).mean()
print(f"\nFraction of utterances ≤2 words: {short_utterances:.3f}")

# Speaker distribution
print("\nSpeaker distribution:")
print(df["speaker"].value_counts(normalize=True))


############################################
# 4. Cleaning functions
############################################

def clean_transcript_text(text):
    """
    Cleaning decisions:
    - Remove bracketed transcription notes (e.g. [inaudible])
    - Remove explicit 'inaudible' tokens
    - Normalize contractions split by transcription (i m → i'm)
    - Remove filler words (um, uh)
    - Collapse whitespace
    - Lowercase
    """
    if pd.isna(text):
        return text

    text = text.lower()

    # Remove bracketed notes
    text = re.sub(r"\[.*?\]", "", text)

    # Remove inaudible markers
    text = re.sub(r"\b(inaudible|unintelligible)\b", "", text)

    # Fix common split contractions
    text = re.sub(r"\bi\s+m\b", "i'm", text)
    text = re.sub(r"\bi\s+ll\b", "i'll", text)
    text = re.sub(r"\bwe\s+re\b", "we're", text)
    text = re.sub(r"\bthey\s+re\b", "they're", text)

    # Remove filled pauses
    text = re.sub(r"\b(um|uh|erm|mm)\b", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


############################################
# 5. Apply cleaning
############################################

df["cleaned_text_v2"] = df["cleaned_text"].apply(clean_transcript_text)

# Recompute word counts
df["num_words_v2"] = df["cleaned_text_v2"].str.split().str.len()


############################################
# 6. Post-cleaning diagnostics
############################################

print("\n===== POST-CLEANING DIAGNOSTICS =====")

# Empty after cleaning
empty_after_cleaning = (df["cleaned_text_v2"] == "").mean()
print(f"Fraction empty after cleaning: {empty_after_cleaning:.3f}")

# Artifact re-check
post_artifacts = {}
for name, pattern in ARTIFACT_PATTERNS.items():
    post_artifacts[name] = df["cleaned_text_v2"].str.contains(
        pattern, flags=re.IGNORECASE, regex=True, na=False
    ).mean()

print("\nArtifact rates after cleaning:")
print(pd.Series(post_artifacts))

# Short utterances remain (important for responsiveness analysis)
short_after = (df["num_words_v2"] <= 2).mean()
print(f"\nFraction of utterances ≤2 words after cleaning: {short_after:.3f}")

############################################
# 8. Teacher–Student Interaction Metrics
############################################

print("\n===== TEACHER–STUDENT INTERACTION METRICS =====")

# Ensure proper ordering
df = df.sort_values(["OBSID", "turn_idx"])

# Identify teacher vs student
df["is_teacher"] = df["speaker"].str.contains("teacher", case=False, na=False)
df["is_student"] = df["speaker"].str.contains("student", case=False, na=False)

# Previous turn info
df["prev_speaker"] = df.groupby("OBSID")["speaker"].shift(1)
df["prev_text"] = df.groupby("OBSID")["cleaned_text_v2"].shift(1)

############################################
# A. Turn transitions
############################################

df["teacher_after_student"] = (
    df["is_teacher"] &
    df["prev_speaker"].str.contains("student", case=False, na=False)
)

df["student_after_teacher"] = (
    df["is_student"] &
    df["prev_speaker"].str.contains("teacher", case=False, na=False)
)

print("Fraction teacher turns responding immediately to students:",
      df["teacher_after_student"].mean())

print("Fraction student turns following teacher:",
      df["student_after_teacher"].mean())


############################################
# B. Teacher responsiveness proxies
############################################

# Teacher acknowledgment tokens
ACK_PATTERNS = r"\b(?:yes|yeah|okay|right|mmhmm|uh huh)\b"

df["teacher_acknowledgment"] = (
    df["is_teacher"] &
    df["cleaned_text_v2"].str.contains(ACK_PATTERNS, regex=True, na=False) &
    (df["num_words_v2"] <= 5)
)

print("Fraction of teacher turns with acknowledgment tokens:",
      df["teacher_acknowledgment"].mean())


############################################
# C. Teacher reformulation (lexical overlap)
############################################

STOPWORDS = {
    "the","a","an","is","are","to","of","and","that","it","this",
    "you","we","they","i","in","on","for","with","but"
}

def lexical_overlap(row):
    if not row["is_teacher"] or pd.isna(row["prev_text"]):
        return 0.0

    prev_words = {
        w for w in row["prev_text"].split()
        if w not in STOPWORDS
    }
    curr_words = {
        w for w in row["cleaned_text_v2"].split()
        if w not in STOPWORDS
    }

    if len(prev_words) == 0:
        return 0.0

    return len(prev_words & curr_words) / len(prev_words)

df["teacher_reformulation_overlap"] = df.apply(lexical_overlap, axis=1)

############################################
# D. Question-following behavior
############################################

df["teacher_question"] = (
    df["is_teacher"] &
    df["text"].str.contains(r"\?", regex=True, na=False)
)

df["prev_raw_text"] = df.groupby("OBSID")["text"].shift(1)

df["student_after_question"] = (
    df["is_student"] &
    df["prev_raw_text"].str.contains(r"\?", regex=True, na=False)
)


print("Fraction of student turns following teacher questions:",
      df["student_after_question"].mean())


############################################
# E. Per-lesson aggregation
############################################

interaction_summary = df.groupby("OBSID").agg({
    "teacher_after_student": "mean",
    "teacher_acknowledgment": "mean",
    "teacher_reformulation_overlap": "mean",
    "student_after_question": "mean",
    "num_words_v2": "mean"
}).reset_index()

interaction_summary.rename(columns={
    "teacher_after_student": "pct_teacher_immediate_response",
    "teacher_acknowledgment": "pct_teacher_acknowledgment",
    "teacher_reformulation_overlap": "mean_teacher_reformulation",
    "student_after_question": "pct_students_answering_questions",
    "num_words_v2": "mean_utterance_length"
}, inplace=True)

############################################
# Save interaction metrics
############################################

interaction_summary.to_csv(
    "NCTE_Transcripts/processed/interaction_metrics_by_lesson.csv",
    index=False
)

print("\nSaved interaction metrics to interaction_metrics_by_lesson.csv")



############################################
# 8. Save cleaned dataset
############################################

OUTPUT_PATH = "NCTE_Transcripts/processed/ncte_single_utterances_cleaned.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
