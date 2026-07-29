"""
Personalized Learning Path Recommender
Strategy: TF-IDF + Cosine Similarity (optimized from 63.99 baseline)

Improvements over baseline:
  1. Course name repeated 3x in train features (boosts course-topic signal)
  2. Trigrams (1,3) instead of bigrams — captures longer review phrases
  3. max_features=60K (was 30K) — richer vocabulary
  4. min_df=1 — keeps all course-specific technical terms
  5. float32 + dense matmul — faster CPU computation
"""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_DIR    = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\c215051c-6-Archive 4"
OUTPUT_PATH = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\submission.csv"

BATCH_SIZE  = 500
TOP_K       = 10

# ─────────────────────────────────────────────
print("Loading data...")
train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(f"  Train: {train.shape},  Test: {test.shape}")

# ─────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("Building features...")
# Repeat course name 3x so course-specific terms dominate TF-IDF weights
train["feat"] = (train["Course"].apply(clean) + " ") * 3 + train["Reviews"].apply(clean)
test["feat"]  = test["Reviews"].apply(clean)
print(f"  Sample train feat: {train['feat'].iloc[0][:120]}")

# ─────────────────────────────────────────────
print("\nFitting TF-IDF...")
t0 = time.time()

vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),       # trigrams for richer phrase coverage
    max_features=60_000,      # larger vocab
    sublinear_tf=True,
    min_df=1,                 # keep rare course-specific terms
    max_df=0.95,
    strip_accents="unicode",
    dtype=np.float32,
)

train_tfidf = normalize(vec.fit_transform(train["feat"]), norm="l2")
test_tfidf  = normalize(vec.transform(test["feat"]),      norm="l2")
print(f"  train: {train_tfidf.shape}, test: {test_tfidf.shape}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────
print("\nComputing top-10 (batch sparse multiply)...")
t0 = time.time()

train_idx = train["Index"].values
results   = []
n_test    = test_tfidf.shape[0]

for start in range(0, n_test, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n_test)
    batch = test_tfidf[start:end]

    sim = (batch @ train_tfidf.T).toarray()

    for i, row in enumerate(sim):
        top = np.argpartition(row, -TOP_K)[-TOP_K:]
        top = top[np.argsort(row[top])[::-1]]
        results.append((test["Index"].iloc[start + i], train_idx[top].tolist()))

    if (start // BATCH_SIZE) % 5 == 0:
        pct     = end / n_test * 100
        elapsed = time.time() - t0
        eta     = elapsed / end * (n_test - end) if end > 0 else 0
        print(f"  [{pct:5.1f}%] {end}/{n_test}  elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s")

print(f"  Done. Total: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
print("\nBuilding submission...")
submission = pd.DataFrame(results, columns=["Index", "Index_list"])
submission["Index_list"] = submission["Index_list"].apply(str)

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["Index", "Index_list"]
assert (submission["Index"].values == test["Index"].values).all()
lengths = submission["Index_list"].apply(lambda x: len(eval(x)))
assert (lengths == 10).all()
print(f"  All checks passed. Shape: {submission.shape}")

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")

print("\nPreview vs sample_submission.csv:")
s_idx = sample["Index"].tolist()
print(submission[submission["Index"].isin(s_idx)].reset_index(drop=True).to_string())
print("\nSample:")
print(sample.to_string())
