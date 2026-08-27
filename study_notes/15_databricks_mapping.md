# TOPIC 15: Mapping This Project to Databricks

---

## Overview

You already know Databricks and PySpark. This section maps every component of this project to its Databricks equivalent. Key insight: **MLflow concepts transfer 100% — it's the same API**. The platforms differ, but the thinking is identical.

---

## Component-by-Component Mapping

---

### 1. Data Ingestion

**Current repo:** `src/ingest.py` — downloads AG News from HuggingFace, saves to `data/raw/` as CSV.

**Databricks equivalent:**
```python
# Option A: Auto Loader (streaming ingestion from cloud storage)
df = spark.readStream.format("cloudFiles") \
    .option("cloudFiles.format", "json") \
    .load("s3://my-bucket/raw/news/")

# Option B: Direct HuggingFace → Delta Lake
from datasets import load_dataset
ds = load_dataset("ag_news")
spark_df = spark.createDataFrame(ds["train"].to_pandas())
spark_df.write.format("delta").saveAsTable("mlops.news.raw_train")
```

**Key differences:**
- Data lives in Delta Lake tables (not CSV files)
- Schema enforcement is built into Delta (no need for manual validation)
- Data can be versioned with Delta's Time Travel: `spark.read.option("versionAsOf", 5).table("...")`

---

### 2. Data Validation

**Current repo:** `validate_dataframe()` in `ingest.py` — manual Python checks.

**Databricks equivalent:**
- **Databricks Expectations** (Delta Live Tables): declarative data quality rules
- **Great Expectations** on Databricks: expectation suites run on Spark DataFrames

```python
# Delta Live Tables with expectations
@dlt.table
@dlt.expect_all_or_drop({
    "text_not_null": "text IS NOT NULL",
    "label_valid": "label IN (0, 1, 2, 3)",
    "text_min_length": "LENGTH(text) >= 10"
})
def validated_news():
    return spark.read.table("mlops.news.raw_train")
```

**Key differences:**
- Expectations are declarative SQL — not imperative Python
- DLT tracks pass/fail counts in a metrics dashboard automatically
- Violations can quarantine records to a separate table instead of hard-failing

---

### 3. Data Preprocessing & Feature Engineering

**Current repo:** `src/preprocess.py` — pandas, regex cleaning, sklearn train_test_split.

**Databricks equivalent:**
```python
# PySpark UDF for text cleaning
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def clean_text_udf(text):
    import re
    if not text: return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)
    text = re.sub(r"\b\d+\b", "NUM", text)
    return re.sub(r"\s+", " ", text).strip().lower()

cleaned = raw_df.withColumn("text", clean_text_udf("text"))
```

**Databricks Feature Store:**
```python
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Write features to Feature Store (persistent, versioned)
fe.create_table(
    name="mlops.news.text_features",
    primary_keys=["article_id"],
    df=feature_df,
    description="Cleaned text features for news classification"
)
```

**Key difference:** Feature Store guarantees training-serving consistency — the same feature logic runs at both training and inference time. This SOLVES the training-serving skew bug in this project.

---

### 4. Model Training

**Current repo:** `src/train.py` — scikit-learn TF-IDF + LR on a single machine.

**Databricks equivalent (Option A — same model, distributed data):**
```python
# Spark ML for distributed TF-IDF + LR
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=50000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=1000, regParam=0.01)

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])
model = pipeline.fit(train_df)
```

**Databricks equivalent (Option B — scikit-learn with Pandas API):**
```python
# Use pandas_on_spark (Koalas) for familiar pandas syntax on Spark
train_pdf = train_df.toPandas()  # for smaller datasets
# ... same scikit-learn code as this project
```

**MLflow training (IDENTICAL API):**
```python
# This code is the SAME on Databricks — MLflow is native!
mlflow.set_experiment("/Users/user@company.com/news-classification")
with mlflow.start_run(run_name="tfidf_lr_baseline") as run:
    mlflow.log_params({...})
    model.fit(X_train, y_train)
    mlflow.log_metrics({...})
    mlflow.sklearn.log_model(model, "model")
```

---

### 5. MLflow Experiment Tracking

**Current repo:** Local MLflow server (SQLite backend, local artifact storage).

**Databricks equivalent:** MLflow is natively integrated. No setup needed.

```python
import mlflow

# On Databricks, tracking_uri is set automatically
# No need for: mlflow.set_tracking_uri(...)

mlflow.set_experiment("/Users/user@company.com/news-classification")
# or use workspace path
```

**Key differences:**
| Feature | This project | Databricks MLflow |
|---------|-------------|-------------------|
| Tracking URI | `http://mlflow:5000` | Automatic (Databricks managed) |
| Artifact store | Local filesystem | DBFS or cloud storage |
| Authentication | None | Workspace-level |
| UI | `localhost:5000` | Databricks workspace sidebar |
| Collaboration | Single user | All workspace users |
| Model Registry | Local MLflow | Unity Catalog |

---

### 6. Model Registry

**Current repo:** MLflow Model Registry (stages: None → Staging → Production).

**Databricks equivalent:** **Unity Catalog Model Registry**

```python
# Register model to Unity Catalog
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="mlops.news.news_classifier"  # three-level namespace: catalog.schema.model
)

# Set alias instead of stage (new recommended approach)
client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name="mlops.news.news_classifier",
    alias="Production",
    version=3
)

# Load by alias
model = mlflow.sklearn.load_model("models:/mlops.news.news_classifier@Production")
```

**Key differences:**
- Three-level namespace: catalog.schema.model (vs just model_name)
- Aliases replace stages in newer MLflow/UC
- Governance: Unity Catalog provides row-level security, audit logs, lineage
- Cross-environment: same registry accessible from dev, staging, prod workspaces

---

### 7. Local Orchestration (DVC)

**Current repo:** `dvc.yaml` — DVC pipeline with 5 stages.

**Databricks equivalent:** **Databricks Jobs / Workflows**

```yaml
# Databricks asset bundle (databricks.yml)
resources:
  jobs:
    news_classification_pipeline:
      name: News Classification MLOps Pipeline
      tasks:
        - task_key: ingest
          notebook_task:
            notebook_path: /notebooks/01_ingest
          
        - task_key: preprocess
          depends_on: [{task_key: ingest}]
          notebook_task:
            notebook_path: /notebooks/02_preprocess
          
        - task_key: train
          depends_on: [{task_key: preprocess}]
          python_wheel_task:
            package_name: mlops_pipeline
            entry_point: train
          
        - task_key: evaluate
          depends_on: [{task_key: train}]
          notebook_task:
            notebook_path: /notebooks/04_evaluate
```

**Key differences:**
- Databricks Jobs = DVC stages (same DAG concept)
- Each stage runs on a dedicated cluster (scales independently)
- Built-in retry logic, alerting, scheduling
- No need to manage cache manually (Databricks handles compute)

---

### 8. FastAPI Model Serving

**Current repo:** `src/serve.py` — FastAPI running in a Docker container.

**Databricks equivalent:** **Databricks Model Serving** (formerly Serverless Serving)

```python
# Deploy to Databricks Model Serving (managed endpoint)
# Via UI or REST API:
# POST /api/2.0/serving-endpoints

{
  "name": "news-classifier",
  "config": {
    "served_models": [{
      "model_name": "mlops.news.news_classifier",
      "model_version": "3",
      "workload_size": "Small",  # 1-4 workers
      "scale_to_zero_enabled": True
    }]
  }
}
```

**What Databricks provides:**
- Auto-scaling (scale to zero when no traffic)
- Managed SSL, authentication, rate limiting
- Built-in A/B testing (traffic splitting between model versions)
- Latency and error rate monitoring out-of-the-box
- No Dockerfile to maintain

**Key differences:**
| Feature | This project | Databricks Model Serving |
|---------|-------------|--------------------------|
| Deployment | Docker image | API call to register endpoint |
| Scaling | Manual workers | Auto-scale to zero |
| Auth | None | Unity Catalog access control |
| A/B testing | Not implemented | Built-in traffic splitting |
| Monitoring | Manual | Built-in dashboards |

---

### 9. Docker Containerization

**Current repo:** `Dockerfile` + `docker-compose.yml` — manual container management.

**Databricks equivalent:** Not directly needed.

Databricks manages all infrastructure. You don't write Dockerfiles for model serving or for training jobs. For custom environments, you can build Docker images for Databricks clusters:

```dockerfile
# Custom Databricks runtime (for specific Python versions or libraries)
FROM databricksruntime/standard:latest
RUN pip install evidently==0.4.11
```

But this is rarely needed — most cases are handled with `%pip install` in notebooks or `requirements.txt` in Jobs.

---

### 10. CI/CD

**Current repo:** GitHub Actions `.github/workflows/ci.yml` — 4 jobs.

**Databricks equivalent:** GitHub Actions + **Databricks Asset Bundles (DABs)**

```yaml
# .github/workflows/ci.yml (Databricks version)
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/ -v

  deploy_to_staging:
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Databricks staging
        run: |
          pip install databricks-cli
          databricks bundle deploy --target staging
          databricks bundle run news_classification_pipeline

  deploy_to_prod:
    needs: deploy_to_staging
    if: success()
    steps:
      - name: Deploy to Databricks production
        run: databricks bundle deploy --target production
```

**Key equivalences:**
| This project | Databricks |
|-------------|-----------|
| params.yaml | Databricks bundle variables |
| Docker build + push | `databricks bundle deploy` |
| `python src/train.py` in CI | `databricks bundle run job_name` |
| DockerHub | Databricks registry (internal) |

---

### 11. Monitoring

**Current repo:** `monitoring/monitor.py` — Evidently + manual PSI.

**Databricks equivalent:** **Databricks Lakehouse Monitoring**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog

w = WorkspaceClient()

# Create a monitor on the predictions table
w.quality_monitors.create(
    table_name="mlops.news.predictions",
    inference_log=MonitorInferenceLog(
        problem_type="PROBLEM_TYPE_CLASSIFICATION",
        prediction_col="prediction",
        label_col="label",
        model_id_col="model_version"
    ),
    schedule={"quartz_cron_expression": "0 0 * * *"},  # daily
    assets_dir="/Users/user@company.com/monitoring"
)
```

**What Databricks Lakehouse Monitoring provides:**
- Automatic drift detection on Delta tables
- Statistical tests: PSI, KS, Jensen-Shannon divergence
- Drift dashboards in the Databricks UI
- Alerting integration with Slack/Teams/PagerDuty
- No manual feature extraction needed

---

## Concept Transfer Map

These concepts transfer IDENTICALLY between this project and Databricks:

| Concept | Transfers? | Notes |
|---------|-----------|-------|
| MLflow `log_params()`, `log_metrics()`, `log_model()` | ✅ 100% | Same API |
| MLflow `start_run()` context manager | ✅ 100% | Same syntax |
| MLflow Model Registry API | ✅ 100% | UC has extra features but same methods |
| `model_uri` formats (`runs:/`, `models:/`) | ✅ 100% | Same format |
| Stratified train/val/test splitting | ✅ 100% | Same concept, PySpark syntax differs |
| Drift monitoring concepts (PSI, KS) | ✅ 100% | Databricks automates what's manual here |
| Quality gates in CI | ✅ 100% | Same pattern |
| Health check / readiness patterns | ✅ 100% | Databricks serving handles automatically |
| Feature drift detection | ✅ 100% | DLM automates it |

---

## What Your Databricks Experience Already Covers

| Your Skill | Maps to This Project |
|------------|---------------------|
| PySpark DataFrames | pandas in preprocess.py |
| Databricks notebooks | src/*.py scripts |
| Delta Lake | data/raw/ and data/processed/ CSV files |
| Databricks Jobs | dvc.yaml pipeline stages |
| Unity Catalog | MLflow Registry concepts |
| Structured Streaming | real-time ingestion (not in this project) |
| Databricks MLflow | src/train.py mlflow.* calls (same code!) |

---

## Key Insight for Interviews

When asked "how would you scale this to production on cloud?", say:

"The core logic stays the same. I'd replace:
- Local CSV files → Delta Lake tables
- DVC stages → Databricks Jobs with dependency ordering
- Local MLflow → Databricks managed MLflow (same API, zero config)
- Local MLflow Registry → Unity Catalog Model Registry
- FastAPI Docker container → Databricks Model Serving endpoint
- Manual monitoring → Databricks Lakehouse Monitoring
- Drift scripts → Delta Live Tables with Expectations

The MLflow training code is copy-paste compatible — same `mlflow.log_params()`, `mlflow.sklearn.log_model()`, `mlflow.register_model()`. The platform changes; the MLOps thinking doesn't."
