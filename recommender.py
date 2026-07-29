"""
Personalized Learning Path Recommender
Strategy: Course prediction + TF-IDF (v4)

Key insight:
  Train features = course name + review (strong signal)
  Test features  = review only (missing course name → weaker signal)

Fix: predict course for each test review using a fast classifier,
     then inject predicted course into test features.
     This closes the feature gap between train and test.

Score history:
  v1 baseline (TF-IDF bigrams 30K):                   63.99
  v2 optimized (TF-IDF trigrams 60K course boost x3):  67.57
  v3 BM25:                                             55.68  (worse, reverted)
  v4 course prediction + TF-IDF (this):                TBD
"""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

DATA_DIR    = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\c215051c-6-Archive 4"
OUTPUT_PATH = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\submission.csv"

BATCH_SIZE  = 500
TOP_K       = 10

# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────
print("Loading data...")
train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(f"  Train: {train.shape},  Test: {test.shape}")

# ─────────────────────────────────────────────
# 2. Clean
# ─────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ─────────────────────────────────────────────
# 3. Predict course for test reviews
# ─────────────────────────────────────────────
print("\nStep 1: Predicting course for each test review...")
t0 = time.time()

clf_vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    dtype=np.float32,
)
X_train_clf = clf_vec.fit_transform(train["Reviews"].apply(clean))
X_test_clf  = clf_vec.transform(test["Reviews"].apply(clean))

clf = LogisticRegression(
    max_iter=1000,
    C=5.0,
    solver="saga",
    n_jobs=-1,
    multi_class="multinomial",
)
clf.fit(X_train_clf, train["Course"])

train_acc = clf.score(X_train_clf, train["Course"])
test_pred_courses = clf.predict(X_test_clf)

print(f"  Classifier train accuracy : {train_acc:.3f}")
print(f"  Sample predictions        : {test_pred_courses[:5].tolist()}")
print(f"  Done [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────
# 4. Build features — both train and test now have course name
# ─────────────────────────────────────────────
print("\nStep 2: Building TF-IDF features...")
t0 = time.time()

# Train: true course name 3x + review
train["feat"] = (train["Course"].apply(clean) + " ") * 3 + train["Reviews"].apply(clean)

# Test: predicted course name 3x + review  ← closes the feature gap
test["feat"]  = (pd.Series(test_pred_courses).apply(clean) + " ") * 3 + test["Reviews"].apply(clean)

print(f"  Train sample: {train['feat'].iloc[0][:100]}")
print(f"  Test  sample: {test['feat'].iloc[0][:100]}")

vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=60_000,
    sublinear_tf=True,
    min_df=1,
    max_df=0.95,
    strip_accents="unicode",
    dtype=np.float32,
)

train_tfidf = normalize(vec.fit_transform(train["feat"]), norm="l2")
test_tfidf  = normalize(vec.transform(test["feat"]),      norm="l2")
print(f"  train: {train_tfidf.shape}, test: {test_tfidf.shape}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────
# 5. Top-10 by cosine similarity
# ─────────────────────────────────────────────
print(f"\nStep 3: Computing top-{TOP_K} recommendations...")
t0 = time.time()

train_idx = train["Index"].values
results   = []
n_test    = test_tfidf.shape[0]

for start in range(0, n_test, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n_test)
    batch = test_tfidf[start:end]
    sim   = (batch @ train_tfidf.T).toarray()

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
# 6. Build & validate submission
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
