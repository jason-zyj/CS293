import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, DataCollatorWithPadding
from sklearn.metrics import f1_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
STAGE1_LR = 2e-5
STAGE2_LR = 1e-5
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 3
PATIENCE = 2

data_folder = "NCTE_Transcripts/processed/annotations/"
TRAIN_CSV = data_folder + "sampled_teacher_utt_pseudolabels-generated.csv"
VAL_CSV = data_folder + "agreed_annotations.csv"

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
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        labels = torch.tensor(row[LABEL_COLS].values.astype(np.float32), dtype=torch.float32)

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits

# =========================
# TRAIN / EVAL
# =========================
def evaluate(model, loader, threshold=0.3):
    model.eval()
    all_preds, all_labels = [], []

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

    preds_binary = (preds > threshold).int().numpy()
    labels = labels.int().numpy()

    macro_f1 = f1_score(labels, preds_binary, average="macro")
    per_label_f1 = f1_score(labels, preds_binary, average=None)

    return macro_f1, per_label_f1

def train(model, train_loader, val_loader, lr, epochs, pos_weight=None):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
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

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate_fn(batch):
        batch_enc = collator(batch)
        labels = torch.stack([item["labels"] for item in batch])
        batch_enc["labels"] = labels
        return batch_enc

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Compute class weights for rare labels
    label_means = train_df[LABEL_COLS].mean().values
    pos_weights = torch.tensor((1 - label_means) / label_means, dtype=torch.float32).to(DEVICE)

    model = MultiLabelModel().to(DEVICE)

    # Stage 1: freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("\n=== Stage 1: Train on synthetic data ===")
    train(model, train_loader, val_loader, STAGE1_LR, EPOCHS_STAGE1, pos_weight=pos_weights)

    # Stage 2: unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = True
    print("\n=== Stage 2: Fine-tune on human data only ===")
    human_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train(model, human_loader, val_loader, STAGE2_LR, EPOCHS_STAGE2, pos_weight=pos_weights)

    print("Training complete.")

if __name__ == "__main__":
    main()

'''
Current best (2/26/26):
=== Stage 1: Train on synthetic data ===
/Users/elisabeth/Documents/Schoolwork/CS293/edu/lib/python3.8/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Epoch 1: 100%|████████████████████████████████████████████████████| 250/250 [03:11<00:00,  1.31it/s]

Epoch 1
Train Loss: 0.9790540072917938
Val Macro F1: 0.48750602433775453
Per-label F1: [0.50746269 0.3805668  0.57142857 0.49056604]
Saved best model.
Epoch 2: 100%|████████████████████████████████████████████████████| 250/250 [02:56<00:00,  1.42it/s]

Epoch 2
Train Loss: 0.9737033863067627
Val Macro F1: 0.48750602433775453
Per-label F1: [0.50746269 0.3805668  0.57142857 0.49056604]
Epoch 3: 100%|████████████████████████████████████████████████████| 250/250 [02:51<00:00,  1.46it/s]

Epoch 3
Train Loss: 0.9695705931186676
Val Macro F1: 0.48750602433775453
Per-label F1: [0.50746269 0.3805668  0.57142857 0.49056604]
Early stopping triggered.

=== Stage 2: Fine-tune on human data only ===
/Users/elisabeth/Documents/Schoolwork/CS293/edu/lib/python3.8/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Epoch 1: 100%|██████████████████████████████████████████████████████| 25/25 [09:17<00:00, 22.29s/it]

Epoch 1
Train Loss: 1.2950024509429932
Val Macro F1: 0.3831104199421501
Per-label F1: [0.50746269 0.3805668  0.15384615 0.49056604]
Saved best model.
Epoch 2: 100%|██████████████████████████████████████████████████████| 25/25 [06:34<00:00, 15.77s/it]

Epoch 2
Train Loss: 1.1325255393981934
Val Macro F1: 0.5088645677551215
Per-label F1: [0.50746269 0.3805668  0.65686275 0.49056604]
Saved best model.
Epoch 3: 100%|██████████████████████████████████████████████████████| 25/25 [09:48<00:00, 23.53s/it]

Epoch 3
Train Loss: 1.057856957912445
Val Macro F1: 0.5092451547725371
Per-label F1: [0.50746269 0.3805668  0.65838509 0.49056604]
Saved best model.
Training complete.
'''