import pandas as pd
import os
from tqdm import tqdm

# --------------------
# Config
# --------------------
data_folder = "NCTE_Transcripts/"
INPUT_PATH = data_folder + "processed/ncte_single_utterances_cleaned.csv"
OUTPUT_PATH = data_folder + "processed/annotations/sampled_teacher_utt.csv"

N_SAMPLES = 200
N_BEFORE = 4
N_AFTER = 2
RANDOM_SEED = 42

# --------------------
# Load data
# --------------------
df = pd.read_csv(INPUT_PATH)

# Extract OBSID and utterance index
df["OBSID"] = df["comb_idx"].astype(str).str.split("_").str[0]
df["utt_idx"] = df["comb_idx"].astype(str).str.split("_").str[1].astype(int)

df = df.sort_values(["OBSID", "utt_idx"]).reset_index(drop=True)

# --------------------
# Sample teacher utterances
# --------------------
teacher_df = df[df["speaker"] == "teacher"]

sampled = teacher_df.sample(
    n=min(N_SAMPLES, len(teacher_df)),
    random_state=RANDOM_SEED
)

rows = []

# --------------------
# Build centered context rows
# --------------------
for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
    obsid = row["OBSID"]
    center = row["utt_idx"]

    record = {
        "OBSID": obsid,
        "target_comb_idx": row["comb_idx"],
        "target_text": row["text"],
    }

    # Context window
    for offset in range(-N_BEFORE, N_AFTER + 1):
        ctx = df[
            (df["OBSID"] == obsid) &
            (df["utt_idx"] == center + offset)
        ]

        speaker_col = f"ctx_{offset}_speaker"
        text_col = f"ctx_{offset}_text"

        if len(ctx) == 1:
            record[speaker_col] = ctx.iloc[0]["speaker"]
            record[text_col] = ctx.iloc[0]["text"]
        else:
            record[speaker_col] = ""
            record[text_col] = ""

    rows.append(record)

# --------------------
# Save
# --------------------
out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(out_df)} centered examples to {OUTPUT_PATH}")

