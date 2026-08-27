# TOPIC 13: End-to-End Walkthrough (Interview Answer)

---

## The Scenario

Interviewer: "Walk me through your end-to-end MLOps project."

---

## Full 3–5 Minute Answer

"I built an end-to-end MLOps pipeline for a news classification system. The goal was to classify news articles into four categories — World, Sports, Business, and Sci/Tech — and deploy it as a production-ready API. Let me walk through each component.

---

### Problem & Dataset

The dataset is AG News from HuggingFace — 120,000 labeled news articles across four balanced categories. In the pipeline, I subsample to 10,000 for development iteration speed, controlled by a config parameter.

---

### Configuration Management

Everything is driven by a single `params.yaml` file — hyperparameters, dataset choice, MLflow URI, model accuracy thresholds, monitoring thresholds. No values are hardcoded. Every script reads from this file at startup, which means you can reproduce any experiment by checking out the git commit that has the config from that run.

---

### Data Ingestion & Validation

The first pipeline stage is `src/ingest.py`. It downloads AG News from HuggingFace, applies stratified subsampling to keep class balance, and then runs a validation suite before saving to disk. The validation checks: schema (required columns present), null ratio (fails if > 2%), text length (warns on very short texts), class imbalance (warns if > 5x ratio), and duplicate detection. I validate both the train and test splits separately. Hard failures raise immediately — soft issues get logged as warnings. The output is `data/raw/train_raw.csv` and `data/raw/test_raw.csv`.

---

### Preprocessing

`src/preprocess.py` reads the raw CSVs, applies a text cleaning pipeline — HTML removal, URL normalization to 'URL' token, number normalization to 'NUM' token, lowercase — then performs a stratified three-way split: 70% train, 15% validation, 15% test. The split uses `random_state=42` for reproducibility. One thing I'd improve in production: the `clean_text()` function isn't part of the sklearn Pipeline, which means raw text goes into the serving pipeline without cleaning — a training-serving skew issue.

---

### Training & Experiment Tracking

`src/train.py` runs three experiment configurations: a TF-IDF + Logistic Regression baseline with unigrams, a bigram LR variant with higher regularization, and a TF-IDF + LinearSVM variant with calibrated probabilities for confidence scores. Each run is tracked in MLflow inside an experiment called 'news-classification-mlops'. For each run, I log: all hyperparameters, validation metrics (accuracy, F1-macro, precision, recall, AUC-ROC), a classification report, a confusion matrix plot, and the full serialized sklearn Pipeline as an MLflow artifact.

After comparing all three runs by validation accuracy, the best model is evaluated on the held-out test set. If it exceeds the 0.88 accuracy threshold from config, it's registered to the MLflow Model Registry as 'news_classifier', automatically transitioning through Staging to Production.

---

### Model Serving

`src/serve.py` is a FastAPI application that loads the Production model from MLflow Registry at startup — with a fallback to a local pkl file if MLflow is unavailable. It exposes three main endpoints: `POST /predict` for single-text classification returning label, confidence, top-k predictions, model version, and latency; `POST /predict/batch` for up to 100 texts at once; and `GET /health` for health monitoring. Every prediction response includes the model version so you can trace exactly which model version made a specific prediction.

---

### Containerization

The application is containerized with a multi-stage Dockerfile. Stage 1 installs all dependencies including C compiler tools needed to build scikit-learn. Stage 2 copies only the compiled packages, discarding the build tools, producing a lean ~500MB production image. Docker Compose orchestrates three services: the MLflow tracking server, the FastAPI model server, and optional one-shot training and monitoring containers activated via profiles.

---

### Monitoring

`monitoring/monitor.py` detects data drift. Because TF-IDF text is not directly amenable to statistical distribution testing, I extract six numeric proxy features: text length, word count, average word length, sentence count, uppercase ratio, and digit ratio. These are compared between the training reference data and incoming data using Evidently AI — with a fallback to a manual PSI implementation if Evidently isn't available. If PSI exceeds 0.15, the script prints a drift alert and exits with code 1, which causes CI to fail if monitoring is run as part of the pipeline.

---

### CI/CD

The GitHub Actions workflow has four jobs. Job 1 runs on every push: installs dependencies, generates synthetic training data (to avoid HuggingFace download latency in CI), runs all 71 pytest tests across the three test files. Job 2 runs only on main branch pushes: patches params.yaml to point to localhost MLflow, starts a local MLflow server, runs the full ingest → preprocess → train → evaluate pipeline, checks that accuracy meets threshold. Job 3 builds the Docker image and pushes it to DockerHub, tagged by git SHA for immutability. Job 4 writes a pipeline summary regardless of outcome. The quality gate between jobs is explicit: Job 3 never runs if Job 2 fails, so a model below threshold never gets containerized.

---

### What I'd Improve in Production

Three main gaps: First, training-serving skew — the text cleaning step should be inside the sklearn Pipeline so it's applied identically at both training and inference time. Second, model promotion is fully automated; in production, I'd add a Staging gate with human approval or automated integration tests before Production promotion. Third, the monitoring uses training data as the reference and test data as 'current' — in real production, I'd log incoming API requests and use that as the current data distribution."

---

## 60-Second Version

"I built an end-to-end news classification pipeline. Data comes from HuggingFace's AG News dataset. A DVC-orchestrated pipeline runs ingest with data validation, text preprocessing, and stratified splitting. Training runs three TF-IDF plus scikit-learn model configurations, all tracked in MLflow with parameters, metrics, plots, and model artifacts. The best model by validation accuracy is automatically evaluated on the test set and registered to the MLflow Model Registry — only if it exceeds an 0.88 accuracy threshold. A FastAPI app serves the Production model with predict, health, and batch endpoints. Drift monitoring uses Evidently AI with a PSI fallback on numeric text features. GitHub Actions runs a four-job CI/CD pipeline: tests, training, Docker build and push, and a summary. The whole system is containerized with Docker Compose running MLflow, the API, and optional training and monitoring services."

---

## Key Numbers to Know

| Metric | Value |
|--------|-------|
| Dataset | AG News — 120,000 articles, 4 classes |
| CI sample size | 10,000 articles (max_samples) |
| Accuracy threshold | 0.88 (must exceed to register) |
| PSI drift threshold | 0.15 |
| API batch limit | 100 texts per request |
| Test suite | 71 pytest tests |
| Docker image | ~500MB (multi-stage build) |
| Pipeline stages | 5 (ingest, preprocess, train, evaluate, monitor) |
| CI jobs | 4 (test, train-evaluate, docker, notify) |
| MLflow experiment | "news-classification-mlops" |
| Model name | "news_classifier" |
| Deployment stage | Production |

---

## Frequently Asked Follow-ups

**Q: How would you scale this?**
"Horizontal scaling behind a load balancer — multiple FastAPI replicas. For the model itself, TF-IDF + LR is CPU-bound and very fast (~5ms per request). At very high scale, I'd cache frequent inputs in Redis. For the training pipeline, I'd move to Databricks Jobs for distributed processing."

**Q: How would you roll back a bad model?**
"MLflow Registry stores all versions. I'd use the MlflowClient to transition the previous version back to Production stage. The serving code loads by stage name, not version number, so the next model reload picks up the old version. The `/model/reload` endpoint allows zero-downtime hot-swap."

**Q: What monitoring would you add?**
"Three layers: input monitoring (data quality, feature distributions — already done), prediction monitoring (distribution of outputs — not implemented), and performance monitoring (ground-truth accuracy on recent labeled data — not implemented). I'd also add request/latency monitoring with Prometheus + Grafana."

**Q: How do you ensure reproducibility?**
"Four mechanisms: params.yaml is version-controlled alongside code, `random_state=42` is used everywhere, MLflow logs all params for every run (so any model can be reproduced from its run_id), and DVC would lock data files to specific versions for each experiment."
