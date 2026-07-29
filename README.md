# HCL AMPlified Challenge: Personalized Learning Path Recommender

An NLP-based recommender system designed to suggest personalized learning paths (course sequences) by aligning test reviews with historical course reviews.

## Achievement

| Version | Strategy | Score | Status |
|---|---|---|---|
| v1 | TF-IDF bigrams 30K | `63.99` | Accepted |
| v2 | TF-IDF trigrams 60K + course boost x3 | `67.57` | Accepted |
| v3 | BM25 | `55.68` | Accepted (worse) |
| v4 | Course prediction + TF-IDF | `81.00` | Accepted ✅ |

---

## Strategy & Technical Approach

### v4 — Course Prediction + TF-IDF (Current Best: 81.00)

**Key Insight:**
- Train features = `course name + review` (strong signal)
- Test features = `review only` (missing course name → weaker signal)
- **Fix:** Predict the course for each test review, then inject it into test features — closes the feature gap between train and test.

**Pipeline:**

1. **Course Prediction (Step 1)**
   - Train a `LogisticRegression` classifier (`C=5.0, solver=saga, multinomial`) on TF-IDF bigram features of train reviews
   - Achieves **100% train accuracy** — reviews are clearly course-distinct
   - Predict course label for every test review

2. **Feature Construction (Step 2)**
   - Train: `predicted_course × 3 + review` (true course name repeated 3×)
   - Test: `predicted_course × 3 + review` (predicted course name repeated 3×)
   - Both sides now have the same feature structure

3. **TF-IDF Vectorization (Step 3)**
   - `ngram_range=(1, 3)` — trigrams for richer phrase coverage
   - `max_features=60,000` — larger vocabulary
   - `min_df=1` — keeps rare course-specific technical terms
   - `sublinear_tf=True`, L2-normalized

4. **Similarity & Retrieval (Step 4)**
   - Batch cosine similarity (batch size = 500)
   - `numpy argpartition` for efficient top-10 extraction per test review

---

### v2 — Optimized TF-IDF (Score: 67.57)
- Course name repeated 3× in train features
- Trigrams (1–3), 60K features, min_df=1
- Test features: review only (no course name)

### v1 — Baseline TF-IDF (Score: 63.99)
- Course name + review as train features (bigrams, 30K features)
- Review-only test features

---

## Repository Structure

```
├── recommender.py          # Main pipeline — course prediction + TF-IDF
├── recommender.ipynb       # Jupyter notebook version
├── validate.py             # Format/shape check validation helper
├── submission.csv          # Generated predictions (10,977 test items)
├── source_code.zip         # Zipped source for portal upload
└── README.md               # Documentation
```

> Note: Dataset folder `c215051c-6-Archive 4` omitted due to size constraints.

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn
```

### Execution

```bash
python recommender.py
```

Or open `recommender.ipynb` in Jupyter and run all cells.

### Expected Output
- `submission.csv` — 10,977 rows, 2 columns (`Index`, `Index_list`)
- Each `Index_list` contains exactly 10 recommended train indices
- Runtime: ~15 min on CPU

---

## Results Progression

```
63.99  →  67.57  →  55.68  →  81.00
  v1        v2        v3        v4 ✅
```

Target: **90+**
