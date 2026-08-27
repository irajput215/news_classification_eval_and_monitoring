# TOPIC 10: Data Drift & Model Monitoring

---

## 1. What is Data Drift?

Data drift is when the statistical distribution of real-world input data changes from what the model was trained on. The model's code hasn't changed. The model weights haven't changed. But the DATA coming in is different.

**Types of drift:**
1. **Data Quality Drift**: Null values, schema changes, invalid values appear
2. **Feature/Covariate Drift**: Input feature distributions shift (X changes)
3. **Label/Prior Drift**: Distribution of target labels shifts (P(Y) changes)
4. **Concept Drift**: The relationship between X and Y changes (P(Y|X) changes)

---

## 2. Data Quality Drift

**What it is:** The incoming data degrades in quality — nulls appear, text gets corrupted, new encoding issues.

**Examples:**
- A news feed API starts returning `null` for the text field in some records
- A different character encoding causes garbled text

**This repo's approach:**
- `validate_dataframe()` in `ingest.py` catches nulls and schema changes
- `monitoring/monitor.py` uses `DataQualityPreset` from Evidently to monitor ongoing data quality

---

## 3. Feature/Covariate Drift

**What it is:** The DISTRIBUTION of input features changes. For a news classifier, this means the statistical properties of incoming text change.

**Example (AG News):**
```
Training data (2023): 
  avg text length = 250 chars
  word count = 45 words
  mostly financial/sports/tech/world topics

Current data (2025):
  avg text length = 180 chars (shorter social media style)
  word count = 30 words  
  new topics emerging (AI news increased significantly)
```

The model was trained on longer, more formal news articles. It now sees shorter, different-style text. The TF-IDF vocabulary has no good representation for the new distribution.

**How to detect:** Compare distributions of input features between reference (training) data and current data.

---

## 4. Concept Drift

**What it is:** The RELATIONSHIP between features and the label changes. Even if the input distribution is the same, what the correct answer is has changed.

**Example:**
```
2022: Text about "Apple earnings" → correctly classified as Business
2025: Text about "Apple Vision Pro AR" → should be Sci/Tech, not Business
  (Apple has shifted from being primarily a consumer company to a tech platform)
```

The words "Apple", "product", "announce" still appear but now mean something different in context.

**Why concept drift is harder:**
- Feature drift is detectable without labels (just look at distributions)
- Concept drift requires labeled examples from recent data to detect (you need to know what the RIGHT answer is now)

**Not implemented in this repository.** Detecting concept drift requires: labeling recent predictions, computing performance on recent labeled data, and monitoring performance over time.

---

## 5. Prediction Drift

**What it is:** The distribution of model OUTPUTS changes, even without knowing if predictions are correct.

**Example:**
```
Training period: 25% of predictions per class (balanced)
Production period: 60% Sports, 10% each of others
```

This could mean:
- Incoming data is genuinely more sports-focused (feature drift)
- The model has developed a spurious bias toward Sports (concept drift)
- Data quality issue is causing incorrect predictions

**Why monitor prediction drift:** You often get outputs BEFORE you get labels. Prediction drift is an early warning signal.

---

## 6. Where is it used in THIS repository?

**File:** `monitoring/monitor.py`

**Approach:** Because you can't directly compute drift on text (it's unstructured), the monitor extracts NUMERIC FEATURES from text, then uses statistical tests on those features.

```python
# monitoring/monitor.py lines 47-70
def extract_text_features(df):
    feat = pd.DataFrame()
    feat["text_length"]     = df["text"].str.len()
    feat["word_count"]      = df["text"].str.split().str.len()
    feat["avg_word_length"] = df["text"].apply(...)
    feat["num_sentences"]   = df["text"].str.count(r"[.!?]+") + 1
    feat["uppercase_ratio"] = df["text"].apply(...)
    feat["digit_ratio"]     = df["text"].apply(...)
    feat["label"]           = df["label"].values
    return feat
```

These 6 numeric proxies capture text distribution changes:
- `text_length`: Shorter/longer texts signal style change
- `word_count`: Complexity change
- `avg_word_length`: Vocabulary shift (simpler/complex)
- `num_sentences`: Structure change
- `uppercase_ratio`: Style change (ALL CAPS headlines)
- `digit_ratio`: More/fewer numbers (financial news vs sports)

---

## 7. Statistical Tests for Drift Detection

### 7.1 PSI (Population Stability Index)

PSI measures how much a distribution has shifted between reference and current.

```
PSI = Σ (P_ref_i - P_cur_i) × ln(P_ref_i / P_cur_i)
```

Where:
- Split the feature into bins (histogram)
- P_ref_i = proportion of reference data in bin i
- P_cur_i = proportion of current data in bin i

**PSI interpretation:**
```
PSI < 0.1:   No significant shift — population is stable
0.1 ≤ PSI < 0.2:  Moderate shift — monitor closely
PSI ≥ 0.2:   Significant shift — investigate and consider retraining
```

**This project uses PSI threshold = 0.15** (in `params.yaml`):
```python
threshold = PARAMS["monitoring"]["drift_threshold"]  # 0.15

for col in numeric_cols:
    score = psi(ref_feat[col].values, cur_feat[col].values)
    results[col] = {"psi": round(score, 4), "drifted": score > threshold}
```

**PSI implementation in the repository:**
```python
def psi(ref, cur, bins=10):
    ref_counts, edges = np.histogram(ref, bins=bins)
    cur_counts, _     = np.histogram(cur, bins=edges)  # same edges!
    
    ref_pct = (ref_counts + 1e-8) / len(ref)  # +1e-8 avoids log(0)
    cur_pct = (cur_counts + 1e-8) / len(cur)
    
    return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
```

**Key note:** Both distributions use the SAME bin edges (`edges` from reference histogram). This ensures the comparison is apples-to-apples.

### 7.2 KS Test (Kolmogorov-Smirnov)

Tests whether two samples come from the same distribution.

```
H0 (null hypothesis): distributions are the same
KS statistic: maximum absolute difference between CDFs
p-value: probability of seeing this difference by chance

If p < 0.05 → reject null → distributions are significantly different
```

**Not explicitly implemented in this repo** (Evidently uses KS internally).

### 7.3 KL Divergence (Kullback-Leibler)

Measures information loss when using distribution Q to approximate P:
```
KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
```

Note: KL divergence is asymmetric (KL(P||Q) ≠ KL(Q||P)) and undefined when Q=0.
PSI is symmetric (similar to Jensen-Shannon divergence) and more stable — that's why it's preferred in practice.

### 7.4 Wasserstein Distance (Earth Mover's Distance)

Measures the minimum "work" needed to transform one distribution into another. Intuitive: how much earth would you need to move to reshape distribution A into distribution B?

More robust to zero-probability bins than KL divergence. Evidently uses it internally.

---

## 8. Evidently AI

Evidently is a Python library for monitoring ML model and data quality. It generates HTML reports.

```python
# monitoring/monitor.py lines 99-111
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric, ...

report = Report(metrics=[
    DataDriftPreset(),            # per-column drift analysis
    DataQualityPreset(),          # data quality (nulls, duplicates, etc.)
    DatasetMissingValuesMetric(), # overall missing values
    DatasetDriftMetric(),         # dataset-level drift summary
    *[ColumnDriftMetric(col) for col in numeric_cols],   # per-column drift
    *[ColumnSummaryMetric(col) for col in numeric_cols], # distribution summaries
])

report.run(reference_data=ref_feat, current_data=cur_feat)
report.save_html("reports/drift_report.html")
```

**Evidently fallback:** If Evidently isn't installed, the code falls back to the manual PSI implementation. This is good defensive programming.

---

## 9. Drift Report Output

```python
# monitoring/monitor.py lines 133-142
summary = {
    "timestamp"         : datetime.utcnow().isoformat(),
    "reference_rows"    : len(reference_df),
    "current_rows"      : len(current_df),
    "n_drifted_columns" : n_drifted,
    "n_total_columns"   : n_total,
    "drift_share"       : round(share, 4),  # fraction of columns drifted
    "dataset_drifted"   : dataset_drifted,  # True/False
    "report_path"       : str(report_path),
}
```

**Exit code:**
```python
sys.exit(1 if drifted else 0)  # non-zero exit lets CI catch drift
```

This is critical for CI integration. If drift is detected in CI, the monitor job fails, alerting the team.

---

## 10. The Complete Drift Detection Flow

```
Reference data: data/processed/train.csv
                    ↓
             extract_text_features()
             → [text_length, word_count, ...]
                    ↓
Current data: data/processed/test.csv (or new production data)
                    ↓
             extract_text_features()
             → [text_length, word_count, ...]
                    ↓
             run_drift_report(reference, current, report_path)
                    ↓
             Evidently Report OR manual PSI
                    ↓
             drift_summary.json + drift_report.html
                    ↓
             check_and_alert(summary)
                    ↓
             Print drift report, exit(1) if drifted
```

---

## 11. When Should a Model Be Retrained? (Full Framework)

```
STEP 1: Is there data quality drift?
   (nulls appearing, schema changed)
   → Yes: Fix the data pipeline FIRST. Retraining on bad data makes things worse.

STEP 2: Is there feature drift? (PSI > 0.15 on key features)
   → No: Nothing actionable for the model.
   → Yes: Continue to Step 3.

STEP 3: Has model PERFORMANCE actually degraded?
   Need: a labeled sample of recent data (minimum ~200 examples)
   Compute: accuracy/F1 on recent labeled data
   → If still >= threshold: drift is happening but not hurting performance yet.
     Monitor more closely. Schedule retraining in next cycle.
   → If performance degraded: Continue to Step 4.

STEP 4: Is the degradation significant and sustained?
   (not just noise — 3+ consecutive evaluation periods below threshold)
   → Yes: Retrain.

STEP 5: Retrain on:
   - Old reference data (to maintain old patterns)
   - New data (to learn new patterns)
   Ratio depends on how fast the concept is drifting.
```

**IMPORTANT INTERVIEW POINT:** PSI > threshold is a TRIGGER to INVESTIGATE, not a trigger to automatically retrain. The decision to retrain requires human judgment or a clear performance metric.

---

## 12. Production Monitoring Architecture

This repo monitors drift MANUALLY (run `python monitoring/monitor.py`). A production system would:

```
Production Traffic
    ↓
Prediction Logs (request + response stored)
    ↓
Scheduled monitoring job (daily/hourly)
    ↓
Compare recent predictions vs reference data
    ↓
If drift detected:
    → Create incident ticket
    → Alert on-call engineer (PagerDuty)
    → Trigger retraining pipeline (GitHub Actions workflow_dispatch)
    ↓
If performance degraded (requires labels):
    → Human review
    → Approve retraining
    → Deploy new model
```

**Tools in production:**
- **Evidently** (used here): Open source, good for batch reports
- **Prometheus + Grafana**: Metrics collection and dashboards
- **Weights & Biases**: Experiment tracking + production monitoring
- **AWS SageMaker Model Monitor**: Managed drift detection
- **Databricks Lakehouse Monitoring**: Built into Databricks platform

---

## 13. Interview Questions

**Q1 (Beginner): What is the difference between feature drift and concept drift?**
- Strong: "Feature drift (covariate shift) means the INPUT distribution changed — the X. For a news classifier, this would be text length becoming shorter, or new topics appearing. Concept drift means the MAPPING from X to Y changed — the relationship itself is different. For example, the word 'Apple' used to predict 'Business', but now predicts 'Sci/Tech' because Apple is now a tech platform company. Feature drift is detectable without labels. Concept drift requires labels to detect."

**Q2 (Intermediate): Why does this project extract numeric text features instead of monitoring the raw text for drift?**
- Strong: "Drift detection algorithms (PSI, KS test) work on numeric distributions. You can compute a histogram of text_length and compare it between reference and current data. You can't compute a histogram of raw text strings. The extracted features (text_length, word_count, etc.) serve as proxy indicators of distribution shift in the text."

**Q3 (Intermediate): Explain PSI and what the threshold of 0.15 means in this project.**
- Strong: "PSI (Population Stability Index) compares two distributions by binning them and computing a symmetric divergence metric. PSI < 0.1 is stable, 0.1-0.2 is moderate shift, > 0.2 is significant. This project uses 0.15, which is between 'monitor' and 'action'. At PSI > 0.15, the monitoring script prints a drift alert and exits with code 1 — triggering a CI failure if run in the pipeline. This is a signal to investigate, not an automatic retraining trigger."

**Q4 (Advanced): This project uses train.csv as reference data and test.csv as current data. What's wrong with this setup in production?**
- Strong: "The test.csv is from the SAME distribution as training data — it's a held-out split of the same AG News dataset. In production, the 'current data' should be ACTUAL production traffic — real incoming news articles from today's feeds. Using test data as 'current' can only detect synthetic drift if you manually inject it. A proper production monitor would log incoming requests to the API, collect them into a 'current data' CSV, and compare them against the training reference distribution."

---

## Must Know
- Feature drift = input distribution changes (detectable without labels)
- Concept drift = X→Y relationship changes (requires labels to detect)
- PSI: < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant
- Drift detection alone does NOT automatically mean retrain
- Must confirm performance degradation before retraining
- Evidently AI generates HTML drift reports

## Should Know
- KS test, KL divergence, Wasserstein distance as alternatives to PSI
- Why numeric proxy features are used for text drift detection
- The 5-step retraining decision framework
- Production monitoring architecture (logs → batch analysis → alerts)

## Nice to Know
- ADWIN (Adaptive Windowing) algorithm for streaming drift detection
- DDM (Drift Detection Method) for concept drift
- Label delay: you often can't detect concept drift until labels arrive (days/weeks later)
- Confounding factors: is drift from the data or a bug in the preprocessing pipeline?
