"""
Personalized Learning Path Recommender
HCL AMPlified Challenge

Approach:
    Two-stage pipeline to recommend the top-10 most similar
    training reviews for each test review.

    Stage 1 - Course Classification:
        A Logistic Regression classifier is trained on TF-IDF
        features of training reviews to predict the course label
        for each test review. This allows the recommender to inject
        the predicted course name into the test feature vector,
        aligning it with the training feature space.

    Stage 2 - TF-IDF Cosine Similarity Retrieval:
        Training features are constructed by prepending the course
        name (repeated 3 times for emphasis) to the review text.
        Test features use the predicted course name in the same way.
        TF-IDF vectors with trigrams are computed and L2-normalised.
        Cosine similarity is computed in batches to retrieve the
        top-10 most similar training entries per test query.

Requirements:
    pip install pandas numpy scikit-learn
"""

import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\c215051c-6-Archive 4"
OUTPUT_PATH = r"c:\Users\rithu\OneDrive\Desktop\HCL AMPlified Challenge\submission.csv"

TOP_K      = 10
BATCH_SIZE = 500
BOOST      = 3   # number of times course name is repeated in feature text


# ── Helper functions ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise text to lowercase, remove non-alphabetic characters,
    and collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(reviews: pd.Series, courses: pd.Series, boost: int) -> pd.Series:
    """
    Construct feature strings by prepending the course name
    (repeated `boost` times) to the cleaned review text.
    """
    return (courses.apply(clean_text) + " ") * boost + reviews.apply(clean_text)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
train  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(f"  Train : {train.shape}")
print(f"  Test  : {test.shape}")
print(f"  Courses: {train['Course'].nunique()} unique")


# ── Stage 1: Course prediction ────────────────────────────────────────────────

print("\nStage 1: Predicting course labels for test reviews...")
t0 = time.time()

clf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    dtype=np.float32,
)

X_train = clf_vectorizer.fit_transform(train["Reviews"].apply(clean_text))
X_test  = clf_vectorizer.transform(test["Reviews"].apply(clean_text))

classifier = LogisticRegression(
    C=5.0,
    max_iter=1000,
    solver="saga",
    multi_class="multinomial",
    n_jobs=-1,
)
classifier.fit(X_train, train["Course"])

train_accuracy    = classifier.score(X_train, train["Course"])
test_pred_courses = classifier.predict(X_test)

print(f"  Classifier training accuracy : {train_accuracy:.4f}")
print(f"  Sample predictions           : {test_pred_courses[:3].tolist()}")
print(f"  Completed in {time.time() - t0:.1f}s")


# ── Stage 2: TF-IDF feature construction ──────────────────────────────────────

print("\nStage 2: Building TF-IDF feature vectors...")
t0 = time.time()

train_features = build_features(train["Reviews"], train["Course"],            BOOST)
test_features  = build_features(test["Reviews"],  pd.Series(test_pred_courses), BOOST)

tfidf_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=60_000,
    sublinear_tf=True,
    min_df=1,
    max_df=0.95,
    strip_accents="unicode",
    dtype=np.float32,
)

train_matrix = normalize(tfidf_vectorizer.fit_transform(train_features), norm="l2")
test_matrix  = normalize(tfidf_vectorizer.transform(test_features),      norm="l2")

print(f"  Train matrix : {train_matrix.shape}")
print(f"  Test  matrix : {test_matrix.shape}")
print(f"  Completed in {time.time() - t0:.1f}s")


# ── Cosine similarity retrieval ───────────────────────────────────────────────

print(f"\nStage 3: Retrieving top-{TOP_K} recommendations...")
t0 = time.time()

train_indices = train["Index"].values
results       = []

for start in range(0, test_matrix.shape[0], BATCH_SIZE):
    end        = min(start + BATCH_SIZE, test_matrix.shape[0])
    batch      = test_matrix[start:end]
    similarity = (batch @ train_matrix.T).toarray()

    for i, scores in enumerate(similarity):
        top_positions = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
        results.append((
            test["Index"].iloc[start + i],
            train_indices[top_positions].tolist()
        ))

    if (start // BATCH_SIZE) % 5 == 0:
        pct     = end / test_matrix.shape[0] * 100
        elapsed = time.time() - t0
        eta     = (elapsed / end) * (test_matrix.shape[0] - end) if end > 0 else 0
        print(f"  [{pct:5.1f}%]  {end}/{test_matrix.shape[0]}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

print(f"  Completed in {time.time() - t0:.1f}s")


# ── Build and validate submission ─────────────────────────────────────────────

print("\nBuilding submission file...")
submission = pd.DataFrame(results, columns=["Index", "Index_list"])
submission["Index_list"] = submission["Index_list"].apply(str)

assert submission.shape[0] == test.shape[0],                        "Row count mismatch"
assert list(submission.columns) == ["Index", "Index_list"],         "Column name mismatch"
assert (submission["Index"].values == test["Index"].values).all(),  "Index order mismatch"

list_lengths = submission["Index_list"].apply(lambda x: len(eval(x)))
assert (list_lengths == TOP_K).all(), "Some rows do not have exactly 10 recommendations"

print(f"  Validation passed. Shape: {submission.shape}")

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
