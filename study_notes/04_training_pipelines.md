# TOPIC 04: Training Pipelines

---

## 1. What is it?

A **training pipeline** is the complete, reproducible sequence of steps that transforms processed data into a trained model. In MLOps, "pipeline" means something specific:

1. **The data pipeline**: ingest → preprocess → feature engineering
2. **The model pipeline**: feature extraction → classifier (this is the sklearn `Pipeline` object)
3. **The experiment pipeline**: run multiple configurations, compare, select best

An sklearn `Pipeline` bundles preprocessing steps + the model into one object. This is critical because it means the EXACT same transformation is applied consistently.

---

## 2. Why do we need it?

**Without a sklearn Pipeline:**
```python
# WRONG — separate objects, easy to misuse at serving time
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_texts)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# At serving:
new_text_tfidf = vectorizer.transform([new_text])  # Must remember to use same vectorizer!
prediction = model.predict(new_text_tfidf)
```
Problem: You save `model.pkl` but forget to save `vectorizer.pkl`. At serving, you transform with a NEW vectorizer → completely different vocabulary → random predictions.

**With a sklearn Pipeline:**
```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression()),
])
pipeline.fit(train_texts, y_train)
pipeline.predict([new_text])  # automatically applies vectorizer first
```
One object, one save, guaranteed consistency.

---

## 3. Where is it used in THIS repository?

**File:** `src/train.py`

**Key functions:**
- `build_pipeline(config)` — constructs sklearn Pipeline
- `compute_metrics(y_true, y_pred, y_prob, label_names)` — evaluates a pipeline
- `run_experiment(run_name, config, train, val, label_names)` — one MLflow run
- `register_best_model(...)` — promotes the winner to MLflow Registry
- `main()` — runs all 3 experiments, finds best, registers

**Experiment configurations (hardcoded in `EXPERIMENT_CONFIGS`):**
```
Run 1: tfidf_lr_baseline  — LR, unigrams, C=1.0, max_features=30k
Run 2: tfidf_lr_bigrams   — LR, bigrams,  C=5.0, max_features=50k
Run 3: tfidf_svm_bigrams  — SVM, bigrams, C=1.0, max_features=50k
```

---

## 4. How does it work in THIS project?

### The sklearn Pipeline:

```python
def build_pipeline(config):
    vectorizer = TfidfVectorizer(
        max_features = config["max_features"],   # limit vocabulary size
        ngram_range  = tuple(config["ngram_range"]),  # unigrams or bigrams
        sublinear_tf = True,   # log(1+tf) instead of raw tf
        min_df       = 2,      # ignore terms appearing in only 1 doc
        strip_accents = "unicode",
        analyzer     = "word",
    )

    if config["model_type"] == "tfidf_lr":
        classifier = LogisticRegression(
            C            = config["C"],          # inverse regularization strength
            max_iter     = config["max_iter"],
            class_weight = "balanced",           # auto-adjust for imbalance
            solver       = "saga",               # efficient for large datasets
            n_jobs       = -1,                   # use all CPU cores
        )
    elif config["model_type"] == "tfidf_svm":
        svc = LinearSVC(C=config["C"], ...)
        classifier = CalibratedClassifierCV(svc, cv=3)  # adds predict_proba

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])
```

### The experiment loop:

```python
results = []
for cfg in EXPERIMENT_CONFIGS:
    run_name = cfg.pop("run_name")
    acc, run_id = run_experiment(run_name, cfg, train, val, label_names)
    results.append({"run_name": run_name, "val_accuracy": acc, "run_id": run_id})

best = max(results, key=lambda r: r["val_accuracy"])
```

### Inside each experiment run:

```
1. mlflow.start_run(run_name=run_name)
2. mlflow.log_params({...})         # log all hyperparameters
3. pipeline.fit(train["text"], train["label"])
4. pipeline.predict(val["text"])
5. compute_metrics(...)             # accuracy, F1, precision, recall, AUC-ROC
6. mlflow.log_metrics(...)          # log val metrics
7. save classification report → mlflow.log_artifact(...)
8. save confusion matrix → mlflow.log_artifact(...)
9. mlflow.sklearn.log_model(pipeline, ...)  # log the entire pipeline
10. joblib.dump(pipeline, f"models/{run_name}.pkl")  # local backup
```

---

## 5. Code Walkthrough

```python
# src/train.py — SVM calibration (lines 84-90)
elif model_type == "tfidf_svm":
    svc = LinearSVC(
        C            = config["C"],
        max_iter     = config["max_iter"],
        class_weight = config.get("class_weight", "balanced"),
    )
    classifier = CalibratedClassifierCV(svc, cv=3)  # adds predict_proba
```

| Concept | Explanation |
|---------|-------------|
| `LinearSVC` | Support Vector Machine — fast for text, doesn't natively output probabilities |
| `CalibratedClassifierCV` | Wraps any classifier to produce calibrated probability estimates |
| `cv=3` | Uses 3-fold cross-validation to learn probability calibration |
| Why needed | `/predict` endpoint returns `confidence` score → needs `predict_proba()` → SVM doesn't have it by default |

```python
# src/train.py — Model logging (lines 212-228)
signature = infer_signature(train["text"].head(5), pipeline.predict(train["text"].head(5)))
input_ex  = train["text"].head(3).tolist()

mlflow.sklearn.log_model(
    sk_model      = pipeline,
    artifact_path = "model",
    signature     = signature,
    input_example = input_ex,
    registered_model_name = None,  # register best only, later
)

# Local backup
joblib.dump(pipeline, model_dir / f"{run_name}.pkl")
```

| Parameter | Why it exists |
|-----------|---------------|
| `signature` | Documents expected input shape and types → MLflow enforces this at serving time |
| `input_example` | Example inputs saved alongside model → documentation + testing |
| `registered_model_name = None` | Only log the model; don't register yet (we register the BEST one after comparing all 3) |
| `joblib.dump` | Local fallback in case MLflow is unavailable at serving time |

---

## 6. Input → Processing → Output

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| Load splits | `data/processed/train.csv`, `val.csv` | `pd.read_csv()` | train/val DataFrames |
| Build pipeline | Config dict | `TfidfVectorizer` + `LogisticRegression/SVM` | Untrained sklearn Pipeline |
| Fit | `train["text"]`, `train["label"]` | `pipeline.fit()` | Trained Pipeline (TF-IDF vocabulary built) |
| Predict on val | `val["text"]` | `pipeline.predict()`, `.predict_proba()` | y_pred, y_prob arrays |
| Compute metrics | y_true, y_pred, y_prob | accuracy, F1, precision, recall, AUC-ROC | Metrics dict |
| Log to MLflow | All of the above | MLflow API calls | Params, metrics, artifacts, model in MLflow |
| Select best | results list | `max(key=val_accuracy)` | best_run_id, best_pipeline |
| Register | best_pipeline + test set | evaluate on test → register → promote | Production model in MLflow Registry |

---

## 7. Production Perspective

**What this repo does:**
- 3 manual experiment configs compared
- Best model selected by val_accuracy
- Threshold gate: must exceed 0.88 to register
- Local pkl backup for serving fallback

**What a production system would add:**
- **Hyperparameter search**: Optuna, Ray Tune, or MLflow's built-in search
- **Cross-validation**: Single train/val split → unstable accuracy estimates
- **Distributed training**: For large models, Spark MLlib or PyTorch DDP
- **Training job isolation**: Each run in its own container/cluster job
- **Model card**: Automated documentation of model capabilities, limitations, training data

**What could fail:**
- MLflow tracking server is down → `mlflow.log_params()` silently fails or crashes
- Out of memory during TF-IDF fit with 50k features → pipeline fails mid-experiment
- Best model barely beats threshold → registers a mediocre model

---

## 8. Important MLOps Concepts

| Concept | Implemented? | How? |
|---------|-------------|------|
| Experiment tracking | ✅ Yes | MLflow runs with params + metrics |
| Model comparison | ✅ Yes | 3 configs compared on val_accuracy |
| Quality gate | ✅ Yes | `accuracy_threshold: 0.88` in params.yaml |
| Artifact logging | ✅ Yes | Reports, confusion matrix, model logged |
| Reproducibility | ✅ Yes | All hyperparams in config, logged to MLflow |
| Hyperparameter search | ❌ Manual | 3 hardcoded configs, not automated search |
| Cross-validation | ❌ No | Single train/val split |

---

## 9. Common Mistakes

1. **Fitting scaler/vectorizer on val/test**: TF-IDF vocabulary MUST be fit on train only → then transform val and test
2. **Saving model and vectorizer separately**: If they get out of sync, predictions are garbage. Use sklearn Pipeline.
3. **Ignoring class_weight**: With imbalanced classes, model predicts majority class → high accuracy but useless
4. **Not logging all parameters**: If you tune C=5 manually and don't log it → you can't reproduce that run
5. **Using val accuracy to make final claims**: Report TEST accuracy. Val accuracy is for selection.
6. **Not calibrating SVM probabilities**: Raw SVM scores are not probabilities → can't use for confidence thresholds

---

## 10. Interview Questions

**Q1 (Beginner): What is an sklearn Pipeline and why is it preferred over separate objects?**
- Strong: "An sklearn Pipeline chains preprocessing and modeling steps into one object. When you call `pipeline.fit(X_train)`, it fits the vectorizer on training data. When you call `pipeline.transform(X_val)` or `pipeline.predict(X_val)`, it applies the SAME vectorizer. This prevents accidental refit on validation data. You serialize one object, load one object — no risk of vectorizer/model mismatch."

**Q2 (Intermediate): Why does this project use CalibratedClassifierCV for LinearSVC?**
- Strong: "LinearSVC produces decision function scores (can be negative, not bounded by 0-1), not probabilities. The serving API needs to return a confidence value. `CalibratedClassifierCV` fits a Platt scaling function using 3-fold CV to map SVM scores to calibrated probabilities. Without it, calling `predict_proba()` on an SVM would throw an error."

**Q3 (Advanced): This project picks the best model by val_accuracy. What are the risks of this approach?**
- Strong: "Three risks: First, val accuracy depends on the specific random split. A different random_state might rank the models differently. More robust would be k-fold cross-validation. Second, accuracy as a sole metric is problematic for imbalanced classes — a model predicting all 'World' could have 90% accuracy if World is 90% of data. F1-macro would be better. Third, val_accuracy and test_accuracy could diverge if the val set is not representative."

---

## Must Know
- sklearn Pipeline bundles feature extraction and classification into one object
- Always fit preprocessing on TRAIN ONLY, transform all splits
- SVM needs calibration to produce probabilities
- Log ALL hyperparameters to MLflow for reproducibility
- class_weight="balanced" is essential for imbalanced datasets

## Should Know
- TF-IDF parameters: max_features, ngram_range, sublinear_tf, min_df
- How CalibratedClassifierCV works (Platt scaling)
- The relationship between regularization C and model complexity
- Why val set accuracy is used for model selection (not test set)

## Nice to Know
- Optuna for automated hyperparameter search
- Ray Tune for distributed hyperparameter search
- MLflow's `mlflow.sklearn.autolog()` for automatic parameter logging
- Bayesian optimization vs grid search vs random search
