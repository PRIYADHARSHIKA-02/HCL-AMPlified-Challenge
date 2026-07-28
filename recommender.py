"""
Personalized Learning Path Recommender
Strategy: TF-IDF on course reviews + cosine similarity
"""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DATA_DIR = r"c:\Users\rithu\OneDrive\Desktop\hcl\c215051c-6-Archive 4"
OUTPUT_PATH = r"c:\Users\rithu\OneDrive\Desktop\hcl\submission.csv"

print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"  Train: {train.shape}, Test: {test.shape}")


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
train["clean_review"] = train["Reviews"].apply(clean_text)
test["clean_review"]  = test["Reviews"].apply(clean_text)


train["feature_text"] = train["Course"].apply(clean_text) + " " + train["clean_review"]
test["feature_text"]  = test["clean_review"]

print("  Sample train feature text:", train["feature_text"].iloc[0][:120])


print("Fitting TF-IDF vectorizer...")
t0 = time.time()

vectorizer = TfidfVectorizer(
    max_features=30_000,       
    ngram_range=(1, 2),        
    sublinear_tf=True,          
    min_df=2,                   
    max_df=0.95,                
    strip_accents="unicode",
    analyzer="word",
)

train_tfidf = vectorizer.fit_transform(train["feature_text"])
test_tfidf  = vectorizer.transform(test["feature_text"])

print(f"  TF-IDF matrix – train: {train_tfidf.shape}, test: {test_tfidf.shape}  [{time.time()-t0:.1f}s]")

train_tfidf = normalize(train_tfidf, norm="l2")
test_tfidf  = normalize(test_tfidf,  norm="l2")


print("Computing top-10 recommendations (batch mode)...")
t0 = time.time()

BATCH_SIZE = 500           
TOP_K = 10
train_indices = train["Index"].values   
results = []  

n_test = test_tfidf.shape[0]
for start in range(0, n_test, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_test)
    batch = test_tfidf[start:end]

    
    sim_matrix = (batch @ train_tfidf.T).toarray()  # shape (batch, n_train)

    for i, row_sim in enumerate(sim_matrix):
        # argsort descending, take top K
        top_k_pos = np.argpartition(row_sim, -TOP_K)[-TOP_K:]
        top_k_pos = top_k_pos[np.argsort(row_sim[top_k_pos])[::-1]]  # sort by sim
        top_k_indices = train_indices[top_k_pos].tolist()
        test_idx = test["Index"].iloc[start + i]
        results.append((test_idx, top_k_indices))

    if (start // BATCH_SIZE) % 5 == 0:
        pct = end / n_test * 100
        print(f"  [{pct:5.1f}%] processed {end}/{n_test} rows  [{time.time()-t0:.1f}s]")

print(f"  Done. Total time: {time.time()-t0:.1f}s")


print("Building submission file...")
submission = pd.DataFrame(results, columns=["Index", "Index_list"])

submission["Index_list"] = submission["Index_list"].apply(lambda x: str(x))

print(f"  Submission shape: {submission.shape}")
print(submission.head(3).to_string())


assert submission.shape[0] == test.shape[0], "Row count mismatch!"
assert list(submission.columns) == ["Index", "Index_list"], "Column name mismatch!"
assert (submission["Index"].values == test["Index"].values).all(), "Index mismatch!"

list_lengths = submission["Index_list"].apply(lambda x: len(eval(x)))
assert (list_lengths == 10).all(), "Some recommendations don't have exactly 10 entries!"

print(f"  All checks passed. List lengths: min={list_lengths.min()}, max={list_lengths.max()}")

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")

print("\nValidating first 5 rows against sample_submission.csv...")
sample_idx = sample["Index"].tolist()
sub_preview = submission[submission["Index"].isin(sample_idx)].reset_index(drop=True)
print("Our output:")
print(sub_preview.to_string())
print("\nSample submission:")
print(sample.to_string())