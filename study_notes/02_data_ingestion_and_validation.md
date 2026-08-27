# TOPIC 02: Data Ingestion & Data Validation

---

## 1. What is it?

**Data Ingestion** is the controlled process of bringing raw data into your pipeline. It is not just "loading a CSV". It means:
- Knowing WHERE data comes from (source)
- Controlling HOW MUCH data enters (sampling)
- Ensuring data arrives in a reproducible way

**Data Validation** is a programmatic check that the ingested data meets expected quality standards BEFORE any ML processing happens. Think of it as a quality gate at the entrance.

---

## 2. Why do we need it?

**Without data validation:**
- A null value in the text column → model crashes during training
- A missing label class → model thinks there are fewer classes than reality
- A changed schema (column renamed upstream) → cryptic KeyError 3 steps later
- Extreme class imbalance → model learns to always predict majority class

The GIGO principle: Garbage In, Garbage Out. Most ML bugs are actually data bugs.

**With validation:**
- Fail fast: catch problems at step 1, not step 4
- Clear error messages: "Missing column 'label'" vs mysterious AttributeError
- Confidence: you know the data meets minimum quality before training

---

## 3. Where is it used in THIS repository?

**File:** `src/ingest.py`

**Functions:**
- `load_ag_news(max_samples)` — Downloads AG News from HuggingFace
- `load_truthlens(csv_path)` — Loads a local CSV
- `validate_dataframe(df, split_name)` — Runs all quality checks
- `save_raw(train_df, test_df)` — Saves to `data/raw/`
- `main()` — Orchestrates the above

**Output:** `data/raw/train_raw.csv` and `data/raw/test_raw.csv`

---

## 4. How does it work in THIS project?

### Step-by-step flow:

1. `main()` reads `PARAMS["data"]["dataset"]` → decides which loader to call
2. `load_ag_news()` downloads from HuggingFace using `datasets` library
3. If `max_samples` is set, it stratified-samples: ensures each class gets `max_samples // 4` rows
4. Label integers (0,1,2,3) are mapped to names: `{0:"World", 1:"Sports", 2:"Business", 3:"Sci/Tech"}`
5. `validate_dataframe()` is called TWICE: once for train, once for test
6. Validation checks: schema, null ratio, text length, class imbalance, duplicates
7. `save_raw()` writes CSVs to `data/raw/`

### The validation function in detail:

```python
REQUIRED_COLUMNS = {"text", "label", "label_name"}
MIN_TEXT_LENGTH  = 10
MAX_NULL_RATIO   = 0.02        # 2%
MAX_CLASS_IMBALANCE = 5.0      # majority / minority <= 5x

def validate_dataframe(df, split_name="train"):
    report = {"split": split_name, "rows": len(df), "issues": []}

    # 1. Schema check — HARD FAILURE
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 2. Null check — HARD FAILURE (>2%)
    null_ratio = df["text"].isnull().mean()
    if null_ratio > MAX_NULL_RATIO:
        raise ValueError(f"Null ratio {null_ratio:.2%} exceeds threshold")

    # 3. Text length — WARNING only
    short_mask = df["text"].str.len() < MIN_TEXT_LENGTH
    n_short = short_mask.sum()
    if n_short > 0:
        report["issues"].append(f"{n_short} texts shorter than 10 chars")

    # 4. Class imbalance — WARNING only
    counts = df["label"].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > MAX_CLASS_IMBALANCE:
        report["issues"].append(f"Class imbalance {imbalance_ratio:.1f}x")

    # 5. Duplicates — WARNING only
    n_dups = df["text"].duplicated().sum()
    if n_dups > 0:
        report["issues"].append(f"{n_dups} duplicate texts")

    return report
```

**Important design decision:** Schema checks and null checks RAISE (hard failures). Length/imbalance/duplicate checks WARN (soft failures). This is intentional — some duplicates may be acceptable, but a schema change is never acceptable.

---

## 5. Code Walkthrough

```python
# src/ingest.py — Stratified subsampling (lines 48-57)
if max_samples:
    train_df = (
        train_df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_samples // 4),
                                          random_state=42))
                .reset_index(drop=True)
    )
```

| Line | What it does | Why it exists | Production problem it solves |
|------|-------------|---------------|------------------------------|
| `groupby("label")` | Group by class label | Ensures sampling is per-class | Without this, you might sample 0 examples from rare class |
| `.apply(lambda g: g.sample(...))` | Sample from each group | `max_samples // 4` = equal per class for 4 classes | Prevents accidental class imbalance from subsampling |
| `random_state=42` | Fixed random seed | Reproducibility | Same sample every run → comparable experiments |
| `.reset_index(drop=True)` | Clean 0-based index | Avoids index-based bugs later | Without this, pandas operations on index can give wrong results |

```python
# src/ingest.py — Null check (lines 105-110)
null_ratio = df["text"].isnull().mean()
report["null_ratio"] = round(null_ratio, 4)
if null_ratio > MAX_NULL_RATIO:
    raise ValueError(
        f"[{split_name}] Null ratio {null_ratio:.2%} exceeds threshold {MAX_NULL_RATIO:.2%}"
    )
```

| Line | What it does | Why it exists |
|------|-------------|---------------|
| `.isnull().mean()` | Fraction (0-1) of null values | Percentage is more meaningful than count |
| `round(null_ratio, 4)` | 4 decimal places | Stored in report dict for logging |
| `raise ValueError(...)` | Hard failure | Null text → model can't train → fail loudly now |
| Formatted error message | Shows actual vs threshold | Gives the operator enough info to diagnose |

---

## 6. Input → Processing → Output

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| Download | HuggingFace AG News API | `load_dataset("ag_news")` | Raw DataFrames |
| Subsampling | Full dataset | Stratified sample per class | Balanced subset |
| Label mapping | Integer labels (0-3) | Dict map | Human-readable label_name column |
| Schema validation | DataFrame | Column set comparison | Pass or ValueError |
| Null validation | text column | `.isnull().mean()` | Pass or ValueError |
| Text length check | text column | `.str.len() < 10` | Warning in report |
| Imbalance check | label column | `.value_counts()` ratio | Warning in report |
| Duplicate check | text column | `.duplicated().sum()` | Warning in report |
| Save | Clean DataFrames | `.to_csv()` | `data/raw/train_raw.csv`, `data/raw/test_raw.csv` |

---

## 7. Production Perspective

**What this repo implements:**
- Schema validation (column presence check)
- Null ratio check with threshold
- Class imbalance check
- Duplicate detection
- Stratified sampling for reproducibility

**What a production system would add:**
- **Data lineage tracking**: Which source, which timestamp, which version?
- **Great Expectations or Deequ**: Declarative expectation suites instead of manual checks
- **Alerting**: Send Slack/PagerDuty alert on validation failure, not just print
- **Data catalog integration**: Register the dataset in a catalog (AWS Glue, Unity Catalog)
- **Idempotency**: Running ingest twice gives same result (this repo does this via `random_state`)
- **Schema evolution handling**: What if a new class is added to the source?

**What could fail:**
- HuggingFace is down → `load_dataset("ag_news")` throws ConnectionError
- Source data schema changes → missing column → ValueError (caught here, good!)
- Disk full → `.to_csv()` fails silently or writes partial file

---

## 8. Important MLOps Concepts Behind This Implementation

| Concept | Implemented? | How? |
|---------|-------------|------|
| Reproducibility | ✅ Yes | `random_state=42` in sampling |
| Idempotency | ✅ Yes | Same config → same output every time |
| Data versioning | ❌ No | DVC tracks file hashes but no version history |
| Validation | ✅ Yes | `validate_dataframe()` with thresholds |
| Schema enforcement | ✅ Yes | REQUIRED_COLUMNS check |
| Lineage | ❌ Partial | Logs source but doesn't track in registry |
| Failure recovery | ❌ No | If download fails mid-way, no retry/resume |

---

## 9. Common Mistakes

1. **Training on unvalidated data**: Skipping validation → null propagation → silent model degradation
2. **Non-stratified subsampling**: `df.sample(1000)` → might get 990 samples from class 0, 10 from others
3. **Loading test data into training**: A classic data leakage bug — combining then splitting
4. **Not resetting index**: After `.groupby().apply()`, index can be duplicated — causes subtle bugs
5. **Hard-failing on warnings**: Treating "10 duplicate texts" as fatal → rejects valid data
6. **Soft-failing on hard errors**: Logging "schema missing column" as a warning → downstream crash

---

## 10. Interview Questions

**Q1 (Beginner): What is data validation and why should it happen before training?**
- Testing: Basic understanding of GIGO principle
- Strong: "Data validation catches quality problems at the pipeline entrance. If null values enter training, the model either crashes or learns from incomplete data. By raising an exception early, we fail fast with a clear error message instead of debugging a mysterious crash 3 steps later."
- Weak: "To make sure data is clean."

**Q2 (Intermediate): Why does this project use stratified sampling instead of random sampling?**
- Testing: Understanding of class imbalance
- Strong: "AG News has 4 equal classes. `max_samples=10000` means 2500 per class. Without stratification, random sampling might give 3000 from class 0 and 500 from class 3 by chance. A model trained on that would bias toward class 0. Stratified sampling guarantees equal representation."
- Follow-up: "When would stratified sampling be WRONG to use?"
- Answer: "In time-series data, stratified sampling would break temporal order — you'd have future data in training and past data in test."

**Q3 (Advanced): This repo validates data in ingest.py. What's missing that a production system would require?**
- Strong: "Three things: First, lineage — we log where data came from but don't register it in a data catalog. Second, declarative expectations — instead of custom Python checks, a tool like Great Expectations generates HTML reports and integrates with CI. Third, schema evolution handling — if a new class appears in AG News, this pipeline silently includes it without alerting the model owner."

---

## 11. Connection to Other MLOps Concepts

```
Data Ingestion
    ↓ produces data/raw/
Data Validation (in same file)
    ↓ passes validated data
Preprocessing (src/preprocess.py reads data/raw/)
    ↓
DVC tracks both steps as stages in dvc.yaml
    ↓
If params.yaml changes (dataset or max_samples),
DVC invalidates cache and re-runs ingest
    ↓
CI runs ingest.py as part of train-and-evaluate job
```

---

## Must Know
- Data validation must happen BEFORE training
- Schema checks should hard-fail; statistical checks can soft-fail
- Stratified sampling prevents accidental class imbalance
- `random_state=42` is the basis of reproducibility
- Null values in the text column will crash a TF-IDF vectorizer

## Should Know
- Great Expectations and Deequ as production-grade validation frameworks
- Data lineage and data catalog concepts
- The difference between schema validation and statistical data quality checks
- Why you validate BOTH train and test splits

## Nice to Know
- Apache Atlas for data lineage
- Delta Lake / Iceberg for versioned data tables
- The concept of a "data contract" between upstream producers and your pipeline
