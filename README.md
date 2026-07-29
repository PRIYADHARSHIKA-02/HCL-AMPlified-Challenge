# HCL AMPlified Challenge: Personalized Learning Path Recommender

An NLP-based recommender system that suggests personalized learning paths by aligning test reviews with historical course reviews.

## Final Achievement

| Version | Strategy | Score |
|---|---|---|
| v1 | TF-IDF bigrams 30K | 63.99 |
| v2 | TF-IDF trigrams 60K + course boost x3 | 67.57 |
| v3 | BM25 | 55.68 |
| **v4** | **Course prediction + TF-IDF** | **81.00 ✅ BEST** |
| v5 | Confidence weighted boost | 80.38 |
| v6 | Course-filtered retrieval | 68.32 |
| v7 | Top-3 LinearSVC injection | 64.07 |
| v8 | Combined blend (A+B) | 77.39 |

---

## Best Strategy: v4 — Course Prediction + TF-IDF (81.00)

### Key Insight
Train features = `course name + review` (strong signal)
Test features = `review only` (missing course name → weaker signal)

**Fix:** Predict the course for each test review using a classifier, then inject the predicted course name into test features — closing the feature gap between train and test.

### Pipeline

**Step 1 — Course Prediction**
- Train `LogisticRegression` (`C=5.0, solver=saga, multinomial`) on TF-IDF bigram features
- Achieves **100% train accuracy** — reviews are clearly course-distinct
- Predictions cached to `test_pred_courses.npy` for fast re-runs

**Step 2 — Feature Construction**
- Train: `true_course × 3 + review`
- Test:  `predicted_course × 3 + review` ← closes the feature gap

**Step 3 — TF-IDF Vectorization**
- `ngram_range=(1, 3)` — trigrams for richer phrase coverage
- `max_features=60,000`
- `min_df=1`, `sublinear_tf=True`, L2-normalized

**Step 4 — Similarity & Retrieval**
- Batch cosine similarity (batch=500)
- `numpy argpartition` for efficient top-10 extraction

---

## Repository Structure

```
├── recommender.py              # Final best model (v4 — 81.00)
├── recommender_v4_best_81.py   # v4 backup copy
├── recommender.ipynb           # Jupyter notebook version
├── validate.py                 # Format/shape validation helper
├── submission.csv              # Final predictions (10,977 rows)
├── test_pred_courses.npy       # Cached course predictions
├── source_code.zip             # Zipped source for portal upload
└── README.md                   # This file
```

> Dataset folder `c215051c-6-Archive 4` omitted due to size.

---

## How to Run

```bash
pip install pandas numpy scikit-learn
python recommender.py
```

Runtime: ~15 min on CPU (classifier ~10 min + retrieval ~5 min).
On subsequent runs, classifier is loaded from cache → ~5 min total.

---

## Results Progression

```
63.99 → 67.57 → 81.00 ✅
  v1      v2      v4
```
