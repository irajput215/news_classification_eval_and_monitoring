# TOPIC 05: MLflow — Complete Deep Dive

---

## 1. What is MLflow?

MLflow is an open-source platform for managing the complete machine learning lifecycle. It has 4 components:

1. **Tracking** — Record and compare experiments (params, metrics, artifacts)
2. **Projects** — Package ML code for reproducibility
3. **Models** — Standard format for model serialization and deployment
4. **Registry** — Centralized model store with versioning and lifecycle management

**This project uses:** Tracking + Models + Registry (not Projects in a formal sense).

MLflow does NOT train your model. It OBSERVES and RECORDS what your code does.

---

## 2. Why MLflow Exists

**The problem it solves:**

Without MLflow:
```
Train run 1: accuracy=0.87, C=1.0, features=30k, saved as "model_v1.pkl"
Train run 2: accuracy=0.89, C=5.0, features=50k, saved as "model_v2.pkl"
Train run 3: ??? — you forgot to write it down
Which model is in production? What were its parameters? How do I reproduce it?
```

With MLflow:
```
mlflow.log_params({"C": 1.0, "max_features": 30000})
mlflow.log_metric("val_accuracy", 0.87)
→ Stored permanently in MLflow with run_id=abc123
→ Can reproduce any run by its run_id
→ Can compare all runs in the UI
```

---

## 3. Core Concepts — The MLflow Data Model

```
MLflow Server (tracking_uri = http://mlflow:5000)
└── Experiment: "news-classification-mlops"
    ├── Run: tfidf_lr_baseline (run_id=abc123)
    │   ├── Parameters: {C: 1.0, max_features: 30000, ...}
    │   ├── Metrics: {val_accuracy: 0.87, val_f1_macro: 0.86, ...}
    │   ├── Artifacts:
    │   │   ├── reports/cls_report_tfidf_lr_baseline.txt
    │   │   ├── reports/confusion_matrix_tfidf_lr_baseline.png
    │   │   └── model/
    │   │       ├── MLmodel          ← model metadata
    │   │       ├── model.pkl        ← serialized sklearn Pipeline
    │   │       ├── input_example.json
    │   │       └── conda.yaml / python_env.yaml
    ├── Run: tfidf_lr_bigrams (run_id=def456)
    └── Run: tfidf_svm_bigrams (run_id=ghi789)   ← BEST

Model Registry:
└── news_classifier
    └── Version 1
        ├── Created from run: ghi789
        ├── Stage: Production
        └── Description: "val_acc=0.91 test_acc=0.90"
```

---

## 4. Experiment Tracking in THIS Repository

### Setup (src/train.py lines 332-338):
```python
tracking_uri = PARAMS["mlflow"]["tracking_uri"]  # "http://mlflow:5000"
exp_name     = PARAMS["mlflow"]["experiment_name"]  # "news-classification-mlops"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(exp_name)
```

**What `set_tracking_uri` does:** Points MLflow client at the server. All subsequent API calls send data to this server.

**What `set_experiment` does:** Creates the experiment if it doesn't exist. All subsequent runs belong to this experiment.

---

## 5. Runs

A **Run** is one execution of your training code. Each run is isolated and identified by a unique `run_id`.

```python
# src/train.py lines 169-229
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id    # e.g., "a1b2c3d4e5f6..."
    
    # Everything inside this context manager is part of this run
    mlflow.log_params(...)
    mlflow.log_metrics(...)
    mlflow.sklearn.log_model(...)
```

The `with` statement is a **context manager**. When the block exits:
- If successful → run status = "FINISHED"
- If exception → run status = "FAILED"
- This status is visible in the MLflow UI

**Accessing run metadata OUTSIDE the context:**
```python
# src/train.py lines 261-262
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
```
This reopens an EXISTING run (by `run_id`) to add test metrics after training is complete.

---

## 6. Parameters

Parameters are **hyperparameters and configuration values** — anything that describes HOW the model was built. They're immutable once logged.

```python
# src/train.py lines 174-184
mlflow.log_params({
    "model_type"  : config["model_type"],    # "tfidf_lr"
    "max_features": config["max_features"],  # 30000
    "ngram_range" : str(config["ngram_range"]),  # "[1, 1]"
    "C"           : config["C"],             # 1.0
    "max_iter"    : config["max_iter"],       # 1000
    "class_weight": config.get("class_weight", "balanced"),
    "dataset"     : PARAMS["data"]["dataset"],  # "ag_news"
    "train_size"  : len(train),              # 7000
    "val_size"    : len(val),                # 1500
})
```

**Key point:** Notice `"ngram_range": str(config["ngram_range"])`. MLflow params must be strings or numbers. A list `[1, 2]` would fail → converted to string `"[1, 2]"`.

**Why log dataset params?** If someone changes the dataset to TruthLens, you can see that in the run — without it, you'd see different metrics and not know why.

---

## 7. Metrics

Metrics are **measured outcomes** of the run — numbers that describe model performance. Unlike parameters, metrics can be logged multiple times (step-based logging for neural networks).

```python
# src/train.py line 195
mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
# Logs: val_accuracy, val_f1_macro, val_precision, val_recall, val_auc_roc

# After finding best model (lines 261-262)
mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
# Logs: test_accuracy, test_f1_macro, test_precision, test_recall, test_auc_roc
```

**Naming convention `val_` and `test_` prefix:** Allows filtering in the UI. You can compare "val_accuracy" across all 3 runs to find the best.

---

## 8. Artifacts

Artifacts are **files** associated with a run — models, plots, reports, datasets.

```python
# src/train.py lines 205, 210
mlflow.log_artifact(str(report_path), artifact_path="reports")
mlflow.log_artifact(str(cm_path), artifact_path="reports")
```

**What gets logged:**
- `reports/cls_report_{run_name}.txt` — sklearn classification_report text
- `reports/confusion_matrix_{run_name}.png` — confusion matrix plot
- `model/` directory — the entire sklearn pipeline

**Artifact storage:** In this project, artifacts go to `--default-artifact-root ./mlruns/artifacts` (local filesystem). In production, you'd point to S3 or Azure Blob.

---

## 9. Model Logging & Serialization

```python
# src/train.py lines 213-222
signature = infer_signature(
    train["text"].head(5),              # input example
    pipeline.predict(train["text"].head(5))  # output example
)
input_ex = train["text"].head(3).tolist()

mlflow.sklearn.log_model(
    sk_model       = pipeline,
    artifact_path  = "model",     # stored as artifact
    signature      = signature,   # input/output schema
    input_example  = input_ex,    # 3 example inputs
    registered_model_name = None, # don't register now
)
```

**What `infer_signature` does:** Inspects the model input/output and creates a schema:
```
input: {text: string}
output: {0: long, 1: long, 2: long, 3: long}
```
This signature is enforced when the model is served via MLflow serving.

**What gets saved inside the model artifact:**
```
model/
├── MLmodel          ← YAML metadata (flavor, signature, python env)
├── model.pkl        ← the actual serialized Pipeline
├── input_example.json
├── conda.yaml       ← conda environment spec
└── python_env.yaml  ← python environment spec
```

**Model flavors:** MLflow uses "flavors" to abstract model formats:
- `mlflow.sklearn.log_model()` → saves as sklearn flavor
- At load time: `mlflow.sklearn.load_model()` → deserializes back to sklearn Pipeline

---

## 10. Model Registry

The Model Registry is a centralized store for production-ready models. It adds:
- **Named model**: `news_classifier` (not just a run_id)
- **Version numbers**: v1, v2, v3... (auto-incremented)
- **Stages**: None → Staging → Production → Archived
- **Descriptions and tags**

```python
# src/train.py lines 265-293
model_name = PARAMS["mlflow"]["registered_model_name"]  # "news_classifier"
model_uri  = f"runs:/{best_run_id}/model"

# Register creates a new version
mv = mlflow.register_model(model_uri=model_uri, name=model_name)
log.info(f"Registered '{model_name}' version {mv.version}")

# Transition stages
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name=model_name, version=mv.version, stage="Staging"
)
client.transition_model_version_stage(
    name=model_name, version=mv.version, stage="Production"
)
```

**The flow:**
```
run_id: ghi789 → model artifact at runs:/ghi789/model
   ↓  mlflow.register_model()
news_classifier version=1 (stage: None)
   ↓  transition_model_version_stage("Staging")
news_classifier version=1 (stage: Staging)
   ↓  transition_model_version_stage("Production")
news_classifier version=1 (stage: Production)
```

---

## 11. Model Versions

Each time `register_model()` is called, a new version is created. Versions are immutable — once a version is registered, its model artifact cannot change.

```python
# How version is incremented:
# Run 1 registers → version 1
# Run 2 registers → version 2
# version 1 moves to Archived automatically if you promote version 2 to Production
```

Note: This project always promotes straight to Production (Staging is a transient state). In a real workflow, you'd:
1. Register → version N (stage: None)
2. Promote to Staging
3. Run integration tests against Staging endpoint
4. If tests pass, promote to Production
5. Previous Production version → Archived

---

## 12. Loading from the Registry

```python
# src/evaluate.py lines 43-47
model_uri = f"models:/{model_name}/Production"
pipeline  = mlflow.sklearn.load_model(model_uri)

# src/serve.py lines 65-66
model_uri = f"models:/{model_name}/{stage}"  # stage="Production"
pipeline  = mlflow.sklearn.load_model(model_uri)
```

**URI formats:**
- `runs:/{run_id}/model` → loads from a specific training run
- `models:/{model_name}/Production` → loads the current Production version
- `models:/{model_name}/1` → loads a specific version number

Using `models:/{name}/Production` is the CORRECT production pattern — you never hardcode a version number. If you promote v2 to Production, all your services automatically use v2 on next model reload.

---

## 13. Model Stages / Aliases (Production Note)

In MLflow 2.x, **aliases** are replacing stages. The old API:
```python
client.transition_model_version_stage(name, version, stage="Production")
```
Is deprecated in favor of:
```python
client.set_registered_model_alias(name, "production", version)
# Load: mlflow.sklearn.load_model(f"models:/{name}@production")
```

This project uses the old stage API (still works, but will be removed in future MLflow versions). **Important for interviews**: Know both APIs exist.

---

## 14. Reproducibility with MLflow

Given any `run_id`, you can exactly reproduce that model:

```python
# 1. Get the run's parameters
client = mlflow.MlflowClient()
run = client.get_run(run_id)
params = run.data.params
# params = {"C": "1.0", "max_features": "30000", "ngram_range": "[1, 2]", ...}

# 2. Get the exact code version
git_commit = run.data.tags.get("mlflow.source.git.commit")

# 3. Load the exact model
pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

You can also download the exact training data from the artifact store and retrain from scratch.

---

## 15. Rollback

If a new model version underperforms:

```python
# Promote previous version back to Production
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="news_classifier",
    version=1,           # previous version
    stage="Production"
)
# version 2 automatically moves to Staging or stays
```

The serving code `mlflow.sklearn.load_model("models:/news_classifier/Production")` will now load v1 on next reload. The `/model/reload` endpoint in `serve.py` makes this instant without restart.

---

## 16. MLflow Architecture in This Project

```
Code (src/train.py)
    |
    | mlflow.set_tracking_uri("http://mlflow:5000")
    |
    ↓
MLflow Client (Python library)
    |
    | HTTP API calls
    |
    ↓
MLflow Server (docker container: mlops_mlflow)
    |
    ├── Backend Store: SQLite (mlruns/mlflow.db)
    │   Stores: experiments, runs, params, metrics
    │
    └── Artifact Store: Local filesystem (/mlruns/artifacts)
        Stores: model.pkl, plots, reports
    
Docker volume: mlflow_data → /mlruns (persists between container restarts)
```

**In production:** Backend store → PostgreSQL. Artifact store → S3 or GCS.

---

## 17. Complete Flow Diagram

```
src/train.py
    ↓
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("news-classification-mlops")
    ↓
mlflow.start_run(run_name="tfidf_lr_baseline")
    ↓
mlflow.log_params({...})          → saved to SQLite
    ↓
pipeline.fit(train, labels)
    ↓
compute_metrics(val)
    ↓
mlflow.log_metrics({val_accuracy: 0.87})  → saved to SQLite
    ↓
mlflow.log_artifact("confusion_matrix.png")  → saved to /mlruns/artifacts
    ↓
mlflow.sklearn.log_model(pipeline, "model")  → saved to /mlruns/artifacts
    ↓
[run ends — status: FINISHED]
    ↓
[repeat for 2 more configs]
    ↓
[find best by val_accuracy]
    ↓
mlflow.register_model("runs:/{best_run_id}/model", "news_classifier")
    ↓
version=1, stage=None
    ↓
transition_model_version_stage("Staging")
transition_model_version_stage("Production")
    ↓
news_classifier v1 is in Production
    ↓
src/evaluate.py: mlflow.sklearn.load_model("models:/news_classifier/Production")
src/serve.py:    mlflow.sklearn.load_model("models:/news_classifier/Production")
```

---

## 18. Interview Questions

**Q1 (Beginner): What is the difference between an MLflow Experiment and a Run?**
- Strong: "An Experiment is a named group that organizes related runs. In this project, the experiment is 'news-classification-mlops'. Every time we train a model configuration, we create a Run inside that experiment. A Run captures one specific training attempt with its parameters, metrics, and artifacts. You compare runs within an experiment to find the best model."

**Q2 (Intermediate): Why does this project log params with `val_` and `test_` prefixes?**
- Strong: "To distinguish which dataset the metric came from. Both val_accuracy and test_accuracy are logged on the same run. By prefixing, you can filter in the MLflow UI to compare only val_accuracy across runs (for model selection) or only test_accuracy (for final reporting). Without the prefix, you'd have two 'accuracy' metrics and no way to tell them apart."

**Q3 (Intermediate): What is a model signature and why does it matter?**
- Strong: "A signature documents the expected input and output schema of a model — like a type contract. When logged, MLflow stores it in the MLmodel file. If you try to serve the model with wrong input types, MLflow can catch it at the API boundary. In this project, the signature says: input is a list of strings, output is an array of class integers."

**Q4 (Advanced): In this project, models are promoted directly from None to Staging to Production in the same function. What's wrong with this in production?**
- Strong: "It bypasses the purpose of the Staging stage. Staging should be used for: running integration tests against a staging API endpoint, A/B testing the new model against the current production model, getting human approval from a model governance team. Automatically promoting to Production means any model that passes the 0.88 accuracy threshold goes live — no human oversight, no integration testing, no gradual rollout."

**Q5 (Advanced): How would you implement model rollback if a Production model starts failing?**
- Strong: "Two approaches: 1) Manual rollback — use MLflow client to transition the previous version back to Production stage. The serving code loads by stage name, so on next reload it picks up the old version. 2) Automated rollback — monitor model performance, and if accuracy drops below a threshold, trigger a GitHub Actions workflow that calls the MLflow API to roll back. The `/model/reload` endpoint in this project supports hot-reload without restart."

---

## Must Know
- MLflow Tracking: experiments → runs → params + metrics + artifacts
- MLflow Model Registry: named model → versions → stages (None/Staging/Production/Archived)
- `mlflow.sklearn.log_model()` saves the complete sklearn Pipeline
- Loading by stage name (`models:/name/Production`) is the correct production pattern
- Rollback = promote previous version back to Production
- Model signature = input/output type contract

## Should Know
- MLflow URI formats: `runs:/`, `models:/name/stage`, `models:/name/version`
- The MLModel file format and what it contains
- Backend store vs artifact store distinction
- Context manager (`with mlflow.start_run()`) and what it tracks
- MLflow aliases vs stages (new vs old API)

## Nice to Know
- MLflow Projects for packaging reproducible code
- MLflow Deployments plugin for custom deployment targets
- `mlflow.sklearn.autolog()` for automatic logging of all sklearn params/metrics
- MLflow Model Explainability integration with SHAP
