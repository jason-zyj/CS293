import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr

# =========================
# 1. Load Data
# =========================
data_folder = "NCTE_Transcripts/processed/annotations/"
file_path = data_folder + "final_annotations_for_review.csv"

df = pd.read_csv(file_path)

# =========================
# 2. Text Cleaning Function
# =========================
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    # Remove bracketed tags like [inaudible], [laughs], etc.
    text = re.sub(r"\[.*?\]", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    return tokens

# =========================
# 3. Compute Lexical Overlap
# =========================
def lexical_overlap(text1, text2):
    tokens1 = set(clean_text(text1))
    tokens2 = set(clean_text(text2))
    
    if len(tokens1) == 0:
        return 0
    
    overlap = tokens1.intersection(tokens2)
    
    # Proportion of target words reused
    return len(overlap) / len(tokens1)

df["lexical_overlap"] = df.apply(
    lambda row: lexical_overlap(row["target_text"], row["ctx_-1_text"]),
    axis=1
)

# =========================
# 4. Convert Annotations to Numeric
# =========================
annotation_cols = [
    "R1: References prior student content_final",
    "R2: Builds on student content_final",
    "R3: Invites further student thinking_final",
    "C1. No student content available (N/A)_final",
    "C2. Want annotation review_final"
]

def convert_annotation(val):
    if val == "REVIEW":
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

for col in annotation_cols:
    df[col + "_num"] = df[col].apply(convert_annotation)

# =========================
# 5. Correlation Analysis
# =========================
print("\n=== Correlation with Lexical Overlap ===\n")

for col in annotation_cols:
    numeric_col = col + "_num"
    
    valid = df[["lexical_overlap", numeric_col]].dropna()
    
    if len(valid) > 2:
        pearson_corr, p1 = pearsonr(valid["lexical_overlap"], valid[numeric_col])
        spearman_corr, p2 = spearmanr(valid["lexical_overlap"], valid[numeric_col])
        
        print(f"{col}")
        print(f"  Pearson r = {pearson_corr:.4f} (p = {p1:.4f})")
        print(f"  Spearman ρ = {spearman_corr:.4f} (p = {p2:.4f})")
        print(f"  N = {len(valid)}\n")
    else:
        print(f"{col} — Not enough data\n")

# =========================
# 6. Visualization
# =========================

for col in annotation_cols:
    numeric_col = col + "_num"
    valid = df[["lexical_overlap", numeric_col]].dropna()
    
    if len(valid) < 5:
        continue
    
    X = valid["lexical_overlap"].values.reshape(-1, 1)
    y = valid[numeric_col].values
    
    # Compute Pearson r for title
    r, p = pearsonr(valid["lexical_overlap"], valid[numeric_col])
    
    # Logistic regression fit
    model = LogisticRegression()
    model.fit(X, y)
    
    x_range = np.linspace(0, 1, 200).reshape(-1, 1)
    y_prob = model.predict_proba(x_range)[:, 1]
    
    # Jitter y values slightly for visualization
    jitter = np.random.normal(0, 0.02, size=len(y))
    
    plt.figure()
    plt.scatter(valid["lexical_overlap"], y + jitter)
    plt.plot(x_range, y_prob)
    
    plt.xlabel("Lexical Overlap")
    plt.ylabel(col)
    plt.title(f"{col}\nPearson r = {r:.3f} (p={p:.3f})")
    
    plt.ylim(-0.1, 1.1)
    plt.show()

# =========================
# Optional: Save Results
# =========================
# df.to_csv("with_lexical_overlap.csv", index=False)


'''
Output:
=== Correlation with Lexical Overlap ===

R1: References prior student content_final
  Pearson r = 0.3737 (p = 0.0000)
  Spearman ρ = 0.4620 (p = 0.0000)
  N = 176

R2: Builds on student content_final
  Pearson r = -0.0451 (p = 0.5702)
  Spearman ρ = 0.1468 (p = 0.0631)
  N = 161

R3: Invites further student thinking_final
  Pearson r = 0.0176 (p = 0.8267)
  Spearman ρ = 0.0318 (p = 0.6927)
  N = 157

C1. No student content available (N/A)_final
  Pearson r = -0.1858 (p = 0.0147)
  Spearman ρ = -0.2492 (p = 0.0010)
  N = 172

C2. Want annotation review_final
  Pearson r = 0.0141 (p = 0.8531)
  Spearman ρ = 0.0622 (p = 0.4122)
  N = 176
'''
