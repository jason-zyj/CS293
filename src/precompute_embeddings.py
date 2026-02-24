'''
To compute the chat annotations:
python src/precompute_roberta_embeddings.py \
    --input_csv NCTE_Transcripts/processed/annotations/synthetic_annotations.csv \
    --output_embeddings synthetic_embeddings.npz

To compute the human annotations:
python precompute_roberta_embeddings.py \
    --input_csv NCTE_Transcripts/processed/annotations/agreed_annotations.csv \
    --output_embeddings human_embeddings.npz

.npz will contain:
embeddings  → (N, 768)
labels      → (N, 4)
ids         → (N,)
'''
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "roberta-base"
MAX_LEN = 512
BATCH_SIZE = 16

LABEL_COLS = [
    "R1: References prior student content_final",
    "R2: Builds on student content_final",
    "R3: Invites further student thinking_final",
    "C1. No student content available (N/A)_final"
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
# DATASET
# =========================
class NCTEDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def build_input(self, row):
        segments = []

        for i in [-4, -3, -2, -1, 0, 1, 2]:
            speaker_col = f"ctx_{i}_speaker"
            text_col = f"ctx_{i}_text"

            if speaker_col in row and text_col in row:
                text = str(row[text_col])
                if text != "nan":
                    speaker = str(row[speaker_col])
                    segments.append(f"{speaker}: {text}")

        segments.append(f"TARGET: {row['target_text']}")
        return " </s> ".join(segments)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.build_input(row)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        labels = torch.tensor(row[LABEL_COLS].values.astype(float))

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "id": row["target_comb_idx"]
        }


# =========================
# EMBEDDING MODEL
# =========================
class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.mean_pool(outputs.last_hidden_state, attention_mask)


# =========================
# PRECOMPUTE FUNCTION
# =========================
def precompute_embeddings(input_csv, output_path):
    print("Loading data...")
    df = pd.read_csv(input_csv)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = NCTEDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = EmbeddingModel().to(DEVICE)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_ids = []

    print("Computing embeddings...")

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            embeddings = model(input_ids, attention_mask)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch["labels"])
            all_ids.extend(batch["id"])

    embeddings = torch.cat(all_embeddings).numpy()
    labels = torch.cat(all_labels).numpy()
    ids = np.array(all_ids)

    np.savez(
        output_path,
        embeddings=embeddings,
        labels=labels,
        ids=ids
    )

    print(f"\nSaved embeddings to {output_path}")
    print("Embeddings shape:", embeddings.shape)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_embeddings", type=str, required=True)
    args = parser.parse_args()

    precompute_embeddings(args.input_csv, args.output_embeddings)


if __name__ == "__main__":
    main()