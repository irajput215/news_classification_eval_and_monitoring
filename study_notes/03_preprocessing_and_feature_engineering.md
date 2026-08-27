# TOPIC 03: Data Preprocessing & Feature Engineering

---

## 1. What is it?

**Data Preprocessing** transforms raw data into a form the model can learn from. For text: removing noise, normalizing format, filtering low-quality examples.

**Feature Engineering** creates new representations from raw data. For text classification with TF-IDF: the "features" ARE the cleaned text tokens. For drift monitoring: numeric summaries (text length, word count) extracted from text.

**Data Splitting** partitions the dataset into train/validation/test sets using statistical principles to prevent information leakage between splits.

---

## 2. Why do we need it?

**Without preprocessing:**
- `<b>Apple</b> stocks rose` → TF-IDF treats `<b>` and `</b>` as vocabulary words
- URLs like `http://reuters.com/article/...` are treated as meaningful tokens — thousands of unique "words"
- Numbers `42`, `43`, `44`... all treated as separate, distinct tokens despite carrying the same semantic meaning
- Mixed case: "Apple" and "apple" counted as different tokens

**Without proper splitting:**
- Data leakage: if test data influences training (even indirectly), accuracy is artificially inflated
- No validation set: you tune hyperparameters on the test set → the test set is no longer a true holdout

---

## 3. Where is it used in THIS repository?

**File:** `src/preprocess.py`

**Functions:**
- `clean_text(text)` — text normalization
- `stratified_split(df)` — creates train/val/test splits
- `main()` — orchestrates the pipeline

**Output:** `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

---

## 4. How does it work in THIS project?

### Step-by-step flow:

1. Load BOTH raw CSVs (`train_raw.csv` and `test_raw.csv`)
2. **Combine** them back into one DataFrame — important! This ensures val split comes from the same distribution
3. Apply `clean_text()` to every row
4. Filter out any texts < 10 characters
5. Call `stratified_split()` to create train/val/test
6. Save 3 CSV files to `data/processed/`

**Why combine before splitting?** The original AG News "test" split from HuggingFace is already a fixed holdout. This project combines everything and re-splits, giving more control over split sizes and stratification.

### Text cleaning pipeline:
```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Remove HTML tags:   <b>Apple</b>  →  Apple
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Normalize URLs:     http://... → URL
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)
    # 3. Normalize numbers:  "42" → "NUM"
    text = re.sub(r"\b\d+\b", "NUM", text)
    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Lowercase everything
    return text.lower()
```

### Stratified split:
```python
def stratified_split(df):
    test_size = PARAMS["data"]["test_size"]   # 0.15
    val_size  = PARAMS["data"]["val_size"]    # 0.15
    seed      = PARAMS["data"]["random_state"] # 42

    # Step 1: carve out test (15% of total)
    train_val, test = train_test_split(df, test_size=test_size,
                                       stratify=df["label"], random_state=seed)
    # Step 2: carve out val from remaining 85%
    # 0.15 / 0.85 = 0.176 → about 15% of total
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val,
                                  stratify=train_val["label"], random_state=seed)
    return train, val, test
```

---

## 5. Code Walkthrough

```python
# src/preprocess.py — clean_text(), lines 28-40
def clean_text(text: str) -> str:
    if not isinstance(text, str):      # Line 1
        return ""
    text = re.sub(r"<[^>]+>", " ", text)   # Line 2: Remove HTML
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)  # Line 3: URLs
    text = re.sub(r"\b\d+\b", "NUM", text)   # Line 4: Numbers
    text = re.sub(r"\s+", " ", text).strip()  # Line 5: Whitespace
    return text.lower()                # Line 6: Lowercase
```

| Line | What it does | Why it exists | What breaks without it |
|------|-------------|---------------|------------------------|
| `isinstance(text, str)` guard | Handles None/NaN | Null values slipped through validation | `re.sub` on None → TypeError crash |
| `<[^>]+>` → space | Remove HTML tags | News articles often contain HTML | `<br>` etc. become useless vocabulary tokens |
| `URL` replacement | Normalize all URLs to token "URL" | URLs are unique → massive vocabulary bloat | 10,000 unique URL tokens, each seen once |
| `NUM` replacement | Normalize all numbers | "2024" and "2025" have same semantic meaning | Vocabulary explodes with year/number variants |
| `.strip()` | Remove leading/trailing spaces | After regex substitutions, leading spaces possible | TF-IDF might create empty token at beginning |
| `.lower()` | Case normalization | "Apple" ≠ "apple" to TF-IDF without this | Vocabulary doubles with case variants |

```python
# Stratified split — key math (lines 61-64)
relative_val = val_size / (1 - test_size)
# = 0.15 / (1 - 0.15) = 0.15 / 0.85 = 0.1765
train, val = train_test_split(
    train_val, test_size=relative_val, ...
)
```

Why the math? We want val to be 15% of the TOTAL dataset. But we're splitting it from `train_val` which is already 85% of total. So val needs to be `15% / 85% ≈ 17.6%` of `train_val` to equal 15% of total.

---

## 6. Input → Processing → Output

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| Load | `data/raw/train_raw.csv` + `data/raw/test_raw.csv` | `pd.read_csv()`, `pd.concat()` | Single DataFrame (all data) |
| Clean text | Raw text column | 5 regex transformations + lowercase | Normalized text column |
| Filter short | All rows | `df["text"].str.len() >= 10` | Rows with substantive text only |
| Split | Clean DataFrame | Stratified 70/15/15 | train, val, test DataFrames |
| Save | 3 DataFrames | `.to_csv()` | `data/processed/{train,val,test}.csv` |

**Final proportions with 10,000 max_samples:**
- Total: ~10,000 rows
- Train: ~7,000 rows (70%)
- Val: ~1,500 rows (15%)
- Test: ~1,500 rows (15%)

---

## 7. Production Perspective

**What this repo does:**
- Clean text with regex normalization
- Stratified 3-way split
- Consistent with `random_state=42`

**What a production system would add:**
- **Preprocessing pipeline serialization**: The `clean_text()` function must be applied IDENTICALLY at serving time. This repo doesn't serialize the preprocessor — a TF-IDF vectorizer sees clean text, but at serving time, `serve.py` passes raw text directly to the sklearn Pipeline (which includes TF-IDF). The clean_text step is NOT part of the sklearn Pipeline — potential bug!
- **Feature stores**: For complex feature engineering, features are precomputed and stored (Feast, Tecton, Databricks Feature Store)
- **Versioned preprocessing**: If you change `clean_text()`, old models and new data may be incompatible

**Critical weakness of this repo:**
```
Training:  clean_text(raw) → TF-IDF → model
Serving:   raw_text → TF-IDF → model  ← clean_text() is SKIPPED!
```
This is called **training-serving skew**. The model was trained on lowercased, HTML-stripped text, but at serving time it receives uppercase text with HTML. The TF-IDF vocabulary won't match.

**How to fix:** Add `clean_text` as a step in the sklearn Pipeline:
```python
Pipeline([
    ("cleaner", FunctionTransformer(clean_text_vectorized)),
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression()),
])
```

---

## 8. Important MLOps Concepts

| Concept | Implemented? | Notes |
|---------|-------------|-------|
| Stratified splitting | ✅ Yes | Both split steps use `stratify=label` |
| Reproducible splits | ✅ Yes | `random_state=42` |
| Data leakage prevention | ✅ Mostly | Test not seen during training |
| Training-serving skew | ❌ Not addressed | clean_text() not in sklearn Pipeline |
| Feature versioning | ❌ No | CSV files not versioned with DVC remote |

---

## 9. Common Mistakes

1. **Training-serving skew**: Preprocessing in training but NOT at inference time → model degrades in production immediately
2. **Data leakage via split**: Shuffling time-series data → future data in training
3. **Not stratifying**: For imbalanced classes, random split might leave val/test with missing classes
4. **Two-split instead of three**: Using only train/test → you tune hyperparameters on "test" → it's no longer a true holdout
5. **Applying test transforms based on test statistics**: TF-IDF vocabulary should be fit on TRAIN only, then applied to val and test
6. **Forgetting to handle NaN in text**: After preprocessing, `NaN.lower()` throws TypeError

---

## 10. Interview Questions

**Q1 (Beginner): What is training-serving skew and why is it dangerous?**
- Testing: Understanding of the full ML lifecycle
- Strong: "Training-serving skew is when the input to your model during training is different from the input during production. In this project, clean_text() is applied during preprocessing but isn't part of the sklearn Pipeline. So at training time the TF-IDF sees lowercased text without HTML. But at serving time, raw text is passed directly to the Pipeline. The TF-IDF vocabulary was built on cleaned text — so tokens from raw text won't match, reducing accuracy silently."
- Follow-up: "How would you detect this in production?"

**Q2 (Intermediate): Why is a validation set separate from a test set?**
- Strong: "The test set is the final, unbiased measurement of generalization. If you use it to select hyperparameters or choose between models, it leaks information — your model has been implicitly 'tuned' on the test set. The validation set is for hyperparameter search. Once you've selected the best model, you evaluate it ONCE on the test set and report that as your accuracy."

**Q3 (Advanced): How would you handle preprocessing in a feature store context (Databricks)?**
- Strong: "In Databricks, I'd run preprocessing as a Databricks Job, compute features, and write them to the Feature Store. The Feature Store serves features at both training time and inference time — this guarantees no training-serving skew. The feature computation logic is registered once and reused by both the training pipeline and the online serving layer."

---

## Must Know
- Stratified splitting is essential for classification
- Three splits: train (train model), val (select hyperparams), test (final evaluation — touch once)
- Training-serving skew is a critical production failure mode
- `random_state` makes splits reproducible
- Preprocessing must be identical at training and inference time

## Should Know
- The math behind stratified relative split sizes
- How TF-IDF vocabulary is fit on train, applied to val/test
- sklearn Pipeline as a way to bundle preprocessing + model
- Feature stores as a solution to training-serving skew

## Nice to Know
- SMOTE for synthetic oversampling of minority classes
- Text augmentation techniques (back-translation, synonym replacement)
- Feature selection methods for high-dimensional text features
