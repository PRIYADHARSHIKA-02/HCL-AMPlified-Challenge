"""
Personalized Learning Path Recommender
Strategy: Course prediction + TF-IDF (v4 - BEST SCORE: 81.00)

*** THIS IS THE BEST PERFORMING VERSION — DO NOT MODIFY ***

Steps:
  1. Predict course for each test review (LogReg, 100% train accuracy)
  2. Inject predicted course name 3x into test features
  3. TF-IDF trigrams 60K features on all 109K train rows
  4. Batch cosine similarity -> top-10

Score history:
  v1 TF-IDF bigrams 30K:                  63.99
  v2 TF-IDF trigrams 60K course boost x3: 67.57
  v3 BM25:                                55.68
  v4 course prediction flat boost x3:     81.00  <- BEST
  v5 confidence weighted boost:           80.38
  v6 course-filtered retrieval:           68.32
  v7 top-3 LinearSVC injection:           64.07
"""

import pandas as pd
import numpy as np
import re
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

DATA_DIR    = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\c215051c-6-Archive 4"
OUTPUT_PATH = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\submission.csv"
CACHE_PATH  = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\test_pred_courses.npy"

BATCH_SIZE  = 500
TOP_K       = 10
BOOST       = 3    # course name repeated 3x — confirmed best

# ─────────────────────────────────────────────
print("Loading data...")
train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(f"  Train: {train.shape},  Test: {test.shape}")

def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ─────────────────────────────────────────────
print("\nStep 1: Predicting course for each test review...")
t0 = time.time()

if os.path.exists(CACHE_PATH):
    test_pred_courses = np.load(CACHE_PATH, allow_pickle=True)
    print(f"  Loaded from cache [{time.time()-t0:.1f}s]")
else:
    clf_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=30_000,
        sublinear_tf=True, min_df=2,
        strip_accents="unicode", dtype=np.float32,
    )
    X_train_clf = clf_vec.fit_transform(train["Reviews"].apply(clean))
    X_test_clf  = clf_vec.transform(test["Reviews"].apply(clean))
    clf = LogisticRegression(max_iter=1000, C=5.0, solver="saga",
                             n_jobs=-1, multi_class="multinomial")
    clf.fit(X_train_clf, train["Course"])
    train_acc         = clf.score(X_train_clf, train["Course"])
    test_pred_courses = clf.predict(X_test_clf)
    np.save(CACHE_PATH, test_pred_courses)
    print(f"  Train accuracy: {train_acc:.3f}  [{time.time()-t0:.1f}s]")

print(f"  Sample: {test_pred_courses[:3].tolist()}")

# ─────────────────────────────────────────────
print("\nStep 2: Building features...")
train["feat"] = (train["Course"].apply(clean) + " ") * BOOST + train["Reviews"].apply(clean)
test["feat"]  = (pd.Series(test_pred_courses).apply(clean) + " ") * BOOST + test["Reviews"].apply(clean)
print(f"  Boost: {BOOST}x")

# ─────────────────────────────────────────────
print("\nStep 3: Fitting TF-IDF...")
t0 = time.time()
vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3),
    max_features=60_000, sublinear_tf=True,
    min_df=1, max_df=0.95,
    strip_accents="unicode", dtype=np.float32,
)
train_tfidf = normalize(vec.fit_transform(train["feat"]), norm="l2")
test_tfidf  = normalize(vec.transform(test["feat"]),      norm="l2")
print(f"  train: {train_tfidf.shape}, test: {test_tfidf.shape}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────
print(f"\nStep 4: Computing top-{TOP_K}...")
t0 = time.time()
train_idx = train["Index"].values
results   = []
n_test    = test_tfidf.shape[0]

for start in range(0, n_test, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n_test)
    sim   = (test_tfidf[start:end] @ train_tfidf.T).toarray()
    for i, row in enumerate(sim):
        top = np.argpartition(row, -TOP_K)[-TOP_K:]
        top = top[np.argsort(row[top])[::-1]]
        results.append((test["Index"].iloc[start + i], train_idx[top].tolist()))
    if (start // BATCH_SIZE) % 5 == 0:
        pct = end / n_test * 100
        elapsed = time.time() - t0
        eta = elapsed / end * (n_test - end) if end > 0 else 0
        print(f"  [{pct:5.1f}%] {end}/{n_test}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

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
print(f"\nSaved -> {OUTPUT_PATH}")
