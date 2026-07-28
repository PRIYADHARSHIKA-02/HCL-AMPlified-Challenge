# HCL AMPlified Challenge: Personalized Learning Path Recommender

An NLP-based recommender system designed to suggest personalized learning paths (course sequences) by aligning test reviews with historical course reviews.

## Achievement
- **Submission Score:** `63.99000` (Accepted)

---

## Strategy & Technical Approach

The core recommendation strategy leverages text mining and similarity matching:

1. **Text Preprocessing:** 
   - Normalizing text to lowercase.
   - Removing special characters, punctuation, and digits.
   - Collapsing duplicate whitespaces.
2. **Feature Engineering:**
   - Text features are constructed by combining course names with their corresponding reviews.
   - Feature representations are built using a **TF-IDF Vectorizer** (with unigrams and bigrams, sublinear term-frequency scaling, and unicode accent stripping).
   - Features are $L_2$-normalized to yield unit vectors for similarity computation.
3. **Similarity & Retrieval:**
   - Cosine similarity is computed between test queries and train feature matrices in batches to optimize memory footprint.
   - **NumPy Partitioning (`argpartition`)** is used to extract the top-10 closest course matches efficiently for each review, which are then sorted in descending order of similarity.

---

## Repository Structure

```
├── recommender.py          # Main model training and batch recommendation logic
├── validate.py             # Formatter/shape check validation helper
├── submission.csv          # Generated test predictions (10,977 test items)
└── README.md               # Documentation
```

> Note: The dataset folder `c215051c-6-Archive 4` is omitted from this repository due to size constraints.

---

## How to Run

### Requirements
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn
```

### Execution
1. Run the recommendation script to generate predictions:
   ```bash
   python recommender.py
   ```
2. Verify the output format and submission shape using the validation script:
   ```bash
   python validate.py
   ```
