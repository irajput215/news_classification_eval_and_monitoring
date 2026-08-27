# MLOps Study Notes — News Classification Pipeline
## Reverse-Engineered from: github.com/MOHD-OMER/mlops-pipeline

---

## Repository Architecture (Actual — Only What Exists)

```
Raw Data (AG News / TruthLens)
        ↓
  [src/ingest.py]
  Data Ingestion + Validation
        ↓
  [src/preprocess.py]
  Text Cleaning + Stratified Splitting
        ↓
  [src/train.py]
  Model Training (3 runs) + MLflow Logging
        ↓
  [src/train.py — register_best_model()]
  Model Registry (MLflow) + Stage Promotion
        ↓
  [src/evaluate.py]
  Final Test Evaluation
        ↓
  [src/serve.py]
  FastAPI Deployment (Online Inference)
        ↓
  [monitoring/monitor.py]
  Data Drift Monitoring (Evidently + PSI fallback)
        ↓
  [.github/workflows/ci.yml]
  CI/CD (GitHub Actions — 4 jobs)
```

Pipeline Orchestration: dvc.yaml (DVC stages)
Configuration:         params.yaml
Containerization:      Dockerfile + docker-compose.yml
Testing:               tests/ (71 tests across 3 files)

---

## Repository File Map

| MLOps Concept           | File/Directory                          | Purpose                                                  |
|-------------------------|-----------------------------------------|----------------------------------------------------------|
| Configuration Mgmt      | params.yaml                             | Single source of truth for hyperparams, paths, thresholds|
| Data Ingestion          | src/ingest.py                           | Download AG News, validate schema, save to data/raw/     |
| Data Validation         | src/ingest.py (validate_dataframe)      | Null check, class imbalance, text length, duplicates     |
| Data Preprocessing      | src/preprocess.py                       | Clean text, stratified train/val/test split              |
| Feature Engineering     | src/preprocess.py (clean_text)          | HTML removal, URL/number normalization, lowercasing       |
| Model Training          | src/train.py                            | TF-IDF + LR/SVM, 3 hyperparameter runs                  |
| Experiment Tracking     | src/train.py (mlflow.log_params/metrics)| MLflow runs inside experiment                            |
| MLflow Logging          | src/train.py                            | Params, metrics, artifacts, model logging                |
| Model Registry          | src/train.py (register_best_model)      | Register best model to MLflow Registry                   |
| Model Versioning        | MLflow Registry                         | Automatic version increment                              |
| Model Promotion         | src/train.py (transition_model_version_stage) | Staging → Production                              |
| Model Evaluation        | src/evaluate.py                         | Load Production model, evaluate on test set              |
| Model Deployment        | src/serve.py                            | FastAPI REST API: /predict, /health, /model/info         |
| Containerization        | Dockerfile, docker-compose.yml          | Multi-stage Docker build, service orchestration          |
| Data Drift Monitoring   | monitoring/monitor.py                   | Evidently AI + PSI fallback drift detection              |
| Pipeline Automation     | dvc.yaml                                | DVC pipeline with dependency graph                       |
| CI/CD                   | .github/workflows/ci.yml                | 4-job GitHub Actions pipeline                            |
| Testing                 | tests/test_api.py, test_data.py, test_model.py | 71 pytest tests                               |
| Logging                 | All src/ files                          | Python logging with structured format                    |
| Error Handling          | src/serve.py, src/evaluate.py           | Try/except with fallbacks                                |
| Reproducibility         | params.yaml + random_state=42           | Deterministic splits and training                        |

---

## Study Notes Index

| #  | File                                          | Topics Covered                                               |
|----|-----------------------------------------------|--------------------------------------------------------------|
| 01 | 01_project_structure_and_config.md            | ML Project Structure, Configuration Management              |
| 02 | 02_data_ingestion_and_validation.md           | Data Ingestion, Data Validation, Schema Checks              |
| 03 | 03_preprocessing_and_feature_engineering.md   | Text Preprocessing, Feature Engineering, Data Splitting     |
| 04 | 04_training_pipelines.md                      | Training Pipelines, Hyperparameter Runs, scikit-learn Pipelines |
| 05 | 05_mlflow_deep_dive.md                        | MLflow FULL deep dive — all components                      |
| 06 | 06_model_evaluation.md                        | Model Evaluation, Metrics (Acc/F1/AUC/Precision/Recall)    |
| 07 | 07_model_deployment_fastapi.md                | FastAPI Serving, Online/Batch Inference, Health Checks      |
| 08 | 08_docker_containerization.md                 | Docker, Multi-Stage Builds, Docker Compose                  |
| 09 | 09_cicd_github_actions.md                     | CI/CD, GitHub Actions, 4-Job Pipeline Walkthrough           |
| 10 | 10_data_drift_and_monitoring.md               | Data Drift, PSI, KS Test, Evidently AI, Concept Drift      |
| 11 | 11_dvc_data_versioning.md                     | DVC, Pipeline Automation, Data Versioning, Reproducibility  |
| 12 | 12_testing_and_error_handling.md              | pytest, API Tests, Data Tests, Model Tests, Error Handling  |
| 13 | 13_end_to_end_walkthrough.md                  | Full Interview Walkthrough (3-5 min + 60-second version)    |
| 14 | 14_interview_questions.md                     | 30+ Interview Questions (Beginner/Intermediate/Advanced)    |
| 15 | 15_databricks_mapping.md                      | Mapping This Project to Databricks                          |
| 16 | 16_study_roadmap.md                           | Personalized Learning Roadmap + Dependency Graph            |

---

## Quick Reference Cheat Sheet

PIPELINE FLOW:
  params.yaml → ingest.py → preprocess.py → train.py → mlflow registry
                                                             ↓
                                                       evaluate.py
                                                             ↓
                                                       serve.py (FastAPI)
                                                             ↓
                                                       monitor.py (drift)

CI/CD TRIGGER: git push to main →
  Job 1: test   (lint + 71 unit tests)
  Job 2: train-and-evaluate (only on main, not PRs)
  Job 3: docker-build (push to DockerHub)
  Job 4: notify (pipeline summary)

KEY THRESHOLDS (params.yaml):
  accuracy_threshold: 0.88   → model only registered if val_acc >= 0.88
  drift_threshold: 0.15      → PSI > 0.15 triggers drift alert

MLFLOW:
  experiment: "news-classification-mlops"
  model_name: "news_classifier"
  stages: Staging → Production

MODELS COMPARED:
  1. tfidf_lr_baseline   (LR, unigrams, C=1.0)
  2. tfidf_lr_bigrams    (LR, bigrams,  C=5.0)
  3. tfidf_svm_bigrams   (SVM+Calibration, bigrams, C=1.0)
