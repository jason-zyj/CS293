import pandas as pd
import os
from tqdm import tqdm

# --------------------
# Config
# --------------------
data_folder = "NCTE_Transcripts/"
INPUT_PATH = data_folder + "processed/ncte_single_utterances_cleaned.csv"

OLD_SAMPLE_PATH_1 = data_folder + "processed/annotations/sampled_teacher_utt.csv"
OLD_SAMPLE_PATH_2 = data_folder + "processed/annotations/sampled_teacher_utt_pseudolabels.csv"

OUTPUT_PATH = data_folder + "processed/annotations/teacher_utt_for_inference_5_per_transcript.csv"

N_BEFORE = 4
N_AFTER = 2
RANDOM_SEED = 42
N_PER_TRANSCRIPT = 7

# --------------------
# Load full dataset
# --------------------
df = pd.read_csv(INPUT_PATH)

df["OBSID"] = df["comb_idx"].astype(str).str.split("_").str[0]
df["utt_idx"] = df["comb_idx"].astype(str).str.split("_").str[1].astype(int)

df = df.sort_values(["OBSID", "utt_idx"]).reset_index(drop=True)

# --------------------
# Load training utterances to exclude
# --------------------
old_ids = set()

if os.path.exists(OLD_SAMPLE_PATH_1):
    old_sample_1 = pd.read_csv(OLD_SAMPLE_PATH_1)
    old_ids.update(old_sample_1["target_comb_idx"].astype(str))

if os.path.exists(OLD_SAMPLE_PATH_2):
    old_sample_2 = pd.read_csv(OLD_SAMPLE_PATH_2)
    old_ids.update(old_sample_2["target_comb_idx"].astype(str))

print(f"Total training utterances excluded: {len(old_ids)}")

# --------------------
# Filter teacher utterances
# --------------------
teacher_df = df[df["speaker"] == "teacher"].copy()
teacher_df = teacher_df[
    ~teacher_df["comb_idx"].astype(str).isin(old_ids)
]

print(f"Remaining teacher utterances: {len(teacher_df)}")

# --------------------
# Sample 5 per transcript
# --------------------
sampled_list = []

for obsid, group in teacher_df.groupby("OBSID"):
    if len(group) <= N_PER_TRANSCRIPT:
        sampled_list.append(group)
    else:
        sampled_list.append(
            group.sample(n=N_PER_TRANSCRIPT, random_state=RANDOM_SEED)
        )

sampled_df = pd.concat(sampled_list)
sampled_df = sampled_df.sort_values(["OBSID", "utt_idx"]).reset_index(drop=True)

print(f"Total sampled utterances: {len(sampled_df)}")

# --------------------
# Build centered context rows
# --------------------
rows = []

for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    obsid = row["OBSID"]
    center = row["utt_idx"]

    record = {
        "OBSID": obsid,
        "target_comb_idx": row["comb_idx"],
        "target_text": row["text"],
    }

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

print(f"Saved {len(out_df)} inference examples to {OUTPUT_PATH}")