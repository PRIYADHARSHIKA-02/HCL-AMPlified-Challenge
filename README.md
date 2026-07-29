# HCL AMPlified Challenge: Personalized Learning Path Recommender

An NLP-based recommender system designed to suggest personalized learning paths (course sequences) by aligning test reviews with historical course reviews.

## Achievement

| Submission | Score | Status |
|---|---|---|
| Baseline (v1) | `63.99000` | Accepted |
| Optimized (v2) | `67.57000` | Accepted ✅ |

---

## Strategy & Technical Approach

The core recommendation strategy leverages text mining and similarity matching:

### v2 — Optimized TF-IDF (Current Best: 67.57)

1. **Text Preprocessing:**
   - Normalizing text to lowercase.
   - Removing special characters, punctuation, and digits.
   - Collapsing duplicate whitespaces.

2. **Feature Engineering:**
   - Course name repeated **3×** in train features to boost course-topic signal in TF-IDF weights.
   - Test features use review text only (mirrors real test conditions — no course name available).
   - Feature representations built using **TF-IDF Vectorizer** with:
     - **Trigrams (1–3)** instead of bigrams — captures longer review phrases
     - **60,000 max features** (was 30K) — richer vocabulary coverage
     - `min_df=1` — retains rare but course-specific technical terms
     - `sublinear_tf=True`, `strip_accents="unicode"`, `float32` dtype
   - Features are $L_2$-normalized to yield unit vectors for cosine similarity.

3. **Similarity & Retrieval:**
   - Cosine similarity computed between test queries and train feature matrix in batches (batch size = 500).
   - **NumPy `argpartition`** extracts top-10 closest matches efficiently per review, sorted descending by similarity.

### v1 — Baseline TF-IDF (Score: 63.99)

- Course name + review as train features (bigrams, 30K features, min_df=2)
- Review-only test features
- Cosine similarity with batch processing

---

## Repository Structure

```
├── recommender.py          # Main model — optimized TF-IDF recommendation logic
├── recommender.ipynb       # Jupyter notebook version of recommender.py
├── validate.py             # Format/shape check validation helper
├── submission.csv          # Generated test predictions (10,977 test items)
├── source_code.zip         # Zipped source for portal upload
└── README.md               # Documentation
```

> Note: The dataset folder `c215051c-6-Archive 4` is omitted from this repository due to size constraints.

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn
```

### Execution

1. Run the recommendation script to generate predictions:
   ```bash
   python recommender.py
   ```
   Or open and run `recommender.ipynb` in Jupyter.

2. Verify the output format and submission shape:
   ```bash
   python validate.py
   ```

### Expected Output
- `submission.csv` with 10,977 rows
- Each row contains `Index` and `Index_list` (exactly 10 recommended train indices)
- Runtime: ~10 minutes on CPU

---

## Results Progression

```
63.99  →  67.57
 v1          v2
```

Target: **90+**
