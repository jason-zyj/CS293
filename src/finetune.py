import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "roberta-base"
MAX_LEN = 512
BATCH_SIZE = 8
STAGE1_LR = 2e-5
STAGE2_LR = 1e-5
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 3
PATIENCE = 2

data_folder = "NCTE_Transcripts/processed/annotations/"
TRAIN_CSV = data_folder + "sampled_teacher_utt_pseudolabels-generated.csv"   # 2000 ChatGPT labeled 
VAL_CSV = data_folder + "agreed_annotations.csv"           # 200 human annotated

LABEL_COLS = [
    "R1: References prior student content",
    "R2: Builds on student content",
    "R3: Invites further student thinking",
    "C1. No student content available (N/A)"
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

        labels = torch.tensor(
                    row[LABEL_COLS].values.astype(np.float32),
                    dtype=torch.float32
                )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }

# =========================
# MODEL
# =========================
class MultiLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(LABEL_COLS))
        )

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
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits

# =========================
# TRAIN / EVAL
# =========================
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    preds_binary = (preds > 0.5).int().numpy()
    labels = labels.int().numpy()

    macro_f1 = f1_score(labels, preds_binary, average="macro")
    per_label_f1 = f1_score(labels, preds_binary, average=None)

    return macro_f1, per_label_f1


def train(model, train_loader, val_loader, lr, epochs):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        macro_f1, per_label_f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print("Train Loss:", total_loss / len(train_loader))
        print("Val Macro F1:", macro_f1)
        print("Per-label F1:", per_label_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

# =========================
# MAIN
# =========================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_dataset = NCTEDataset(train_df, tokenizer)
    val_dataset = NCTEDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MultiLabelModel().to(DEVICE)

    print("\n=== Stage 1: Train on synthetic data ===")
    train(model, train_loader, val_loader, STAGE1_LR, EPOCHS_STAGE1)

    print("\n=== Stage 2: Fine-tune on human data only ===")
    human_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model, human_loader, val_loader, STAGE2_LR, EPOCHS_STAGE2)

    print("Training complete.")

if __name__ == "__main__":
    main()