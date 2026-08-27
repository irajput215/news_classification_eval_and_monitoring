# TOPIC 14: 30+ Interview Questions — This Project

---

## BEGINNER (10 Questions)

---

**B1. What is the purpose of `params.yaml` in this project?**

Testing: Configuration management fundamentals.

Strong answer: "params.yaml is the single source of truth for all tunable values in the pipeline — hyperparameters, dataset choice, train/val/test split sizes, MLflow tracking URI, accuracy threshold, monitoring threshold. Every script reads from it at startup. Because it's version-controlled in Git, you can reproduce any experiment by checking out the commit that contains the config used at that time. Nothing is hardcoded in source files."

Weak answer: "It stores the model parameters."

Follow-up: "What would happen if a developer hardcoded `C=1.0` in train.py instead of reading from params.yaml?"

---

**B2. What is MLflow and why is it used in this project?**

Testing: Basic MLflow understanding.

Strong answer: "MLflow is an experiment tracking and model lifecycle management platform. In this project, it serves two purposes: tracking (recording every training run's hyperparameters, validation metrics, plots, and model artifact) and the Model Registry (providing a versioned store with staging and production states). Without it, you'd have model_v1.pkl, model_v2.pkl with no record of what hyperparameters produced each. MLflow lets you compare all 3 experiment runs in a UI, select the best, register it with a version number, and load it by name from the serving code."

Follow-up: "What is the difference between an MLflow Run and an MLflow Model Version?"

---

**B3. What does the `/health` endpoint in serve.py do?**

Testing: Production serving basics.

Strong answer: "The /health endpoint returns the service status and model information. If the model is loaded, it returns status: healthy. If it isn't, it returns status: model_not_loaded. In production, load balancers call this endpoint every 30 seconds. If it returns a non-200 response or an unhealthy status, the load balancer removes this instance from the traffic rotation. Docker Compose also uses it to determine when to restart an unhealthy container. The Docker healthcheck in this project calls the /health URL."

Follow-up: "What's the difference between readiness and liveness probes in Kubernetes? How does /health serve both?"

---

**B4. Why does train.py use a sklearn Pipeline instead of fitting a TF-IDF vectorizer and model separately?**

Testing: sklearn Pipeline understanding.

Strong answer: "The Pipeline bundles the vectorizer and classifier into a single object. When you call pipeline.fit(X_train), the vectorizer fits on training data only. When you call pipeline.predict(X_val), the vectorizer transforms val data using the already-fit vocabulary — it does NOT refit. If they were separate, you'd have to remember to call vectorizer.transform (not fit_transform) on val/test. You'd also have to save two separate objects and keep them synchronized. With a Pipeline, you serialize one object and load one object — guaranteed consistency."

Follow-up: "This project has a training-serving skew issue related to the Pipeline. Can you identify it?"

---

**B5. What is the accuracy threshold in this project and why does it exist?**

Testing: Quality gates understanding.

Strong answer: "The accuracy_threshold is set to 0.88 in params.yaml. Before registering a model to the MLflow Registry, the code checks: if val_accuracy < 0.88, the model is NOT registered and the function returns early with a warning. This is a quality gate — a model that doesn't meet minimum quality standards should never reach the production registry. Without it, every training run would automatically promote to production regardless of performance, potentially degrading the system."

Follow-up: "Is validation accuracy the right metric for this gate? What could go wrong?"

---

**B6. What is data drift and why does this project monitor for it?**

Testing: Basic monitoring concepts.

Strong answer: "Data drift is when the statistical distribution of real-world input data changes from the training data the model was built on. The model hasn't changed but the world has. For a news classifier, this could mean the style of news articles changes — shorter social media-style headlines, new topic categories emerging. The model was trained on one distribution and now receives another. This often leads to silent accuracy degradation — the model still returns predictions, just incorrect ones. This project monitors 6 numeric text features (text length, word count, etc.) for drift using PSI. When PSI exceeds 0.15, an alert is triggered."

Follow-up: "What's the difference between feature drift and concept drift?"

---

**B7. Explain the multi-stage Docker build in this project.**

Testing: Docker fundamentals.

Strong answer: "The Dockerfile has two stages. Stage 1 (builder) installs build-essential, gcc, and g++ — needed to compile scikit-learn's C extensions — then runs pip install. Stage 2 (runtime) copies only the compiled Python packages from Stage 1. The build tools (gcc, etc.) are discarded. This produces a final image around 500MB instead of 2GB, reduces the attack surface in production (no compiler in the container), and speeds up deployment."

Follow-up: "Why does the CMD in the Dockerfile use `--host 0.0.0.0` instead of `127.0.0.1`?"

---

**B8. What triggers the GitHub Actions CI/CD pipeline?**

Testing: CI/CD basics.

Strong answer: "The pipeline triggers on three events: push to main or dev branches (if specific paths change — src/, tests/, monitoring/, params.yaml, requirements.txt, or Dockerfile), pull requests to main, and manual dispatch via workflow_dispatch. The path filter is important — pushing a README change doesn't trigger the ML pipeline, only code and config changes do. Manual dispatch also accepts inputs: force_train and accuracy_threshold, allowing ad-hoc training runs with custom thresholds."

Follow-up: "Why does Job 2 (training) only run on the main branch and not on PRs?"

---

**B9. What is DVC and what does it do in this project?**

Testing: Data versioning awareness.

Strong answer: "DVC (Data Version Control) extends Git for ML pipelines. In this project, it primarily provides pipeline automation through dvc.yaml, which defines 5 stages: ingest, preprocess, train, evaluate, monitor. Each stage has deps (source files + input data), params (config values), and outs (output files). When you run dvc repro, DVC checks which stages have changed inputs. If nothing changed, it uses cached results. If params.yaml changes, it invalidates and re-runs only the affected stages. Data versioning with remote storage (S3) is defined in requirements.txt but not configured with a remote in this project."

Follow-up: "What would you need to add to this project to enable team data sharing with DVC?"

---

**B10. How does the serving API load the model, and what happens if MLflow is unavailable?**

Testing: Fault tolerance basics.

Strong answer: "serve.py's _load_model() function tries two paths: first, it connects to MLflow and loads the Production model from the registry using models:/news_classifier/Production. If this succeeds, it also fetches the version number and metrics from MLflow to include in API responses. If MLflow raises any exception, it falls back to loading the most recently sorted .pkl file from the local models/ directory. This graceful degradation means the serving API stays online even if MLflow goes down, at the cost of potentially running an older model version."

Follow-up: "How would you tell which model is running if MLflow loaded the fallback pkl?"

---

## INTERMEDIATE (10 Questions)

---

**I1. Why does this project use `f1_macro` rather than accuracy as the primary metric for model comparison?**

Testing: Metric selection reasoning.

Strong answer: "Accuracy is the fraction of correct predictions across ALL classes. For perfectly balanced datasets (like AG News with equal class sizes), accuracy and macro F1 are nearly equivalent. However, F1-macro computes F1 per class and averages, giving equal weight to each class regardless of size. The project logs both. F1-macro is more informative because it surfaces if one class has poor precision or recall even when overall accuracy looks fine. In production with imbalanced data, F1-macro would be far more important."

Follow-up: "When would weighted-average F1 be preferable to macro F1?"

---

**I2. Explain the stratified split calculation in preprocess.py.**

Testing: Understanding of split arithmetic.

Strong answer: "The code does a two-step split. Step 1 carves out the test set: 15% of total data. This leaves train+val at 85% of total. Step 2 splits val from the remaining: but we want val to be 15% of TOTAL, not 15% of the 85%. So the calculation is relative_val = 0.15 / (1 - 0.15) = 0.15 / 0.85 ≈ 0.176. Taking 17.6% of the 85% gives us exactly 15% of the total. Both splits use stratify=label to ensure each class appears in the same proportion in train, val, and test."

Follow-up: "Why is stratification especially important in the second split?"

---

**I3. This project transitions models from None → Staging → Production automatically in the same function. What's wrong with this in a real production system?**

Testing: Production MLOps maturity.

Strong answer: "Staging exists as a testing environment. A proper workflow would be: register the model as None, then run integration tests against the Staging endpoint (not just the offline evaluation), potentially do A/B traffic splitting, get human approval from a model governance team, then promote to Production. This project skips all of that and promotes automatically. Any model above the 0.88 accuracy threshold on the held-out test set goes live — no integration testing, no canary deployment, no gradual rollout, no human review."

Follow-up: "How would you implement a human approval gate in this GitHub Actions pipeline?"

---

**I4. Why does the monitoring script use numeric proxy features instead of directly comparing text distributions?**

Testing: NLP drift detection understanding.

Strong answer: "Statistical drift tests like PSI and KS compare numeric distributions. You can compute a histogram of text_length values and compare it between reference and current data. You cannot compute a histogram of raw strings. The proxy features — text_length, word_count, avg_word_length, num_sentences, uppercase_ratio, digit_ratio — capture meaningful statistical properties of the text distribution. If the news source starts publishing shorter articles, text_length distribution will shift, triggering a PSI alert. These proxies don't capture semantic drift (change in TOPIC) but they do capture structural and stylistic changes."

Follow-up: "How would you detect topic drift (new categories emerging) without labeled data?"

---

**I5. Why does the CI/CD pipeline patch params.yaml before training? Isn't this a code smell?**

Testing: CI/CD environment management.

Strong answer: "It's a pragmatic workaround for a real limitation. params.yaml has tracking_uri: http://mlflow:5000 — the Docker Compose service name. In GitHub Actions, there's no Docker network, so 'mlflow' doesn't resolve as a hostname. The train.py code reads this URI directly from params.yaml, not from an environment variable, so we can't override it via MLFLOW_TRACKING_URI env var. The correct fix would be to modify train.py to use os.environ.get('MLFLOW_TRACKING_URI', PARAMS['mlflow']['tracking_uri']) — checking env vars first, falling back to config. That would make patching unnecessary."

Follow-up: "Implement the fix in two lines of Python."

---

**I6. Explain how the API returns confidence scores for a LinearSVC model.**

Testing: ML implementation details.

Strong answer: "LinearSVC natively produces decision function scores — distances from the hyperplane — not probabilities. These can be negative and aren't bounded by 0-1. The code wraps LinearSVC with CalibratedClassifierCV(svc, cv=3), which fits a Platt scaling function using 3-fold cross-validation to map SVM decision scores to proper probabilities. After calibration, calling predict_proba() returns a proper probability distribution that sums to 1.0. Without calibration, calling predict_proba() on a LinearSVC raises an AttributeError."

Follow-up: "Is Platt scaling always a good solution? When might it fail?"

---

**I7. How does the GitHub Actions pipeline pass the accuracy value from the evaluate step to the threshold check step?**

Testing: CI/CD data flow.

Strong answer: "The evaluate step uses $GITHUB_OUTPUT to set a step output variable: `echo 'accuracy=${ACCURACY}' >> $GITHUB_OUTPUT`. This file is monitored by the Actions runner. The next step reads it with the expression `${{ steps.evaluate.outputs.accuracy }}`. This is the correct modern approach — older workflows used `::set-output::` which is deprecated. The step also has `id: evaluate` so other steps can reference it by ID."

Follow-up: "What's the difference between step outputs, job outputs, and environment variables in GitHub Actions?"

---

**I8. What does `mlflow.register_model(model_uri, name)` actually do, and what is the `model_uri` format?**

Testing: MLflow Registry mechanics.

Strong answer: "register_model takes a model URI pointing to an artifact logged during a run, and creates a new version in the registry under the given name. The URI format is runs:/{run_id}/model — the run_id is the unique identifier for the MLflow run, and 'model' is the artifact_path used when logging the model with mlflow.sklearn.log_model(). MLflow copies the model artifact from the run's artifact store into the registry, creates a version record in the backend database, and returns a ModelVersion object with the version number. Subsequent calls create version 2, version 3, etc."

Follow-up: "What happens if you call register_model with the same name twice from different runs?"

---

**I9. The test suite deselects `test_no_overlap_train_test` in CI. Is this acceptable? How would you make it pass in CI?**

Testing: Test design thinking.

Strong answer: "It's acceptable as a pragmatic compromise, but it means CI doesn't verify the most critical data integrity property. The fix: generate synthetic CI data with unique texts per class rather than repeating 4 templates. Instead of 4 texts × 300 repetitions, generate 1200 distinct texts using templates with unique suffixes or parameterized variation. This way, the train and test splits would have distinct texts, the overlap test would pass, and CI would verify the full data quality suite."

Follow-up: "Write a 5-line Python snippet to generate 300 unique text variants for one class."

---

**I10. How does Docker Compose ensure the API doesn't start before MLflow is ready?**

Testing: Service orchestration.

Strong answer: "The api service has `depends_on: mlflow: condition: service_healthy`. This tells Docker Compose to wait until the mlflow service passes its healthcheck before starting the api service. The MLflow service has a healthcheck configured: `test: [python -c import urllib.request; urlopen('http://localhost:5000/health')]`, running every 15 seconds, with a 30-second start_period grace period. Compose polls the health status of the mlflow container. Only when it reports healthy does it start the api container. Without this, the api would start immediately, try to connect to MLflow which hasn't finished initializing, fail to load the model, and start in an unhealthy state."

Follow-up: "What are the risks of using service_healthy vs service_started?"

---

## ADVANCED (10 Questions)

---

**A1. This project has a training-serving skew bug. Describe it precisely and fix it.**

Testing: Production ML system awareness.

Strong answer: "The bug: clean_text() is called in preprocess.py during data preparation. The sklearn Pipeline contains only TfidfVectorizer + Classifier. When train.py trains the model, the TF-IDF vocabulary is built on lowercased, HTML-stripped, URL-normalized text. But serve.py passes raw incoming text directly to pipeline.predict_proba() — the clean_text step never runs at inference time. Fix: convert clean_text to work on arrays and add it as a Pipeline step using FunctionTransformer:

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def clean_texts(texts):
    return np.array([clean_text(t) for t in texts])

pipeline = Pipeline([
    ('cleaner', FunctionTransformer(clean_texts)),
    ('tfidf', TfidfVectorizer(...)),
    ('clf', LogisticRegression(...))
])
```

Now the cleaning is baked into the pipeline and applied identically at training and serving."

Follow-up: "After making this fix, do you need to change preprocess.py?"

---

**A2. The project exits with sys.exit(1) if drift is detected. When would you NOT want to do this in production?**

Testing: Nuanced production monitoring design.

Strong answer: "Three scenarios: First, if monitoring runs as a scheduled job (not in CI), exiting with 1 makes the cron job appear failed, which might send false alarms to PagerDuty for what is just an informational alert. Second, if drift occurs regularly due to expected business cycles (sports news spikes during championship seasons), sys.exit(1) would create noisy alerts. Third, in CI, it's appropriate — you want to know that new data differs from training data before deploying. The fix: use a separate exit code (e.g., exit(2) for drift warning, exit(1) for data quality errors) and configure alerting rules accordingly."

---

**A3. How would you implement blue-green deployment for this model API?**

Testing: Advanced deployment patterns.

Strong answer: "Blue-green keeps two identical environments: Blue (current production) and Green (new release). Deployment steps: 1) Deploy new model to Green environment (identical infrastructure, different Docker image tag). 2) Run smoke tests against Green. 3) Switch the load balancer to route 100% traffic from Blue to Green. 4) Keep Blue running for 10 minutes for fast rollback. 5) Tear down Blue after confidence. In Kubernetes: two Deployments with the same labels but different image tags; a Service selector switch routes traffic. In this project, the `/model/reload` endpoint enables an in-place model swap which is simpler but doesn't give zero-downtime for the brief period the model is reloading."

---

**A4. How would you detect concept drift in a production news classifier without labels?**

Testing: Advanced monitoring design.

Strong answer: "Without labels, concept drift is hard to detect directly. Three proxy approaches: First, monitor prediction confidence distribution — if the model's confidence scores drop systematically (average top probability decreasing), it may be struggling with new patterns. Second, monitor prediction distribution shifts — if suddenly 80% of articles are predicted as Sports when historical baseline was 25%, something has changed. Third, implement active learning: sample a small percentage of low-confidence predictions for human labeling, then measure accuracy on that sample over time. This provides a ground truth signal without labeling everything. A performance drop on the labeled sample is direct evidence of concept drift."

---

**A5. The model is registered with `accuracy_threshold: 0.88`. How would you set this threshold in a real project?**

Testing: Model quality criteria.

Strong answer: "The threshold should be set based on: First, business requirements — what accuracy is minimally acceptable for the business use case? For news classification feeding a recommendation engine, maybe 85% is fine; for medical diagnosis, 99.9% might be required. Second, baseline comparison — is 0.88 better than a simple baseline (majority class prediction, or a previous model version)? Third, statistical stability — run the model multiple times with different seeds and check if the typical range is 0.87-0.91. Set the threshold below the typical minimum to allow natural variance. Fourth, cost asymmetry — is a false positive or false negative more costly? If so, use precision or recall threshold instead of accuracy."

---

**A6. How would you scale this system to handle 100,000 requests per minute?**

Testing: System design at scale.

Strong answer: "At 100k RPM (~1,667 RPS): First tier — horizontal pod autoscaling in Kubernetes: start with 5 FastAPI replicas, each handling ~333 RPS; TF-IDF + LR does ~5ms/request, so theoretical 200 RPS per single-core worker; 4 workers × 5 replicas = 3,000+ RPS with headroom. Second tier — caching: news headlines repeat (same headline from multiple sources); add Redis cache with 60-second TTL for identical text inputs, easily achieving 50%+ cache hit rate. Third tier — async batch processing: queue single-text requests and process in micro-batches of 32; saves TF-IDF transform overhead. Fourth tier — CDN for static responses: if the same article appears repeatedly, edge-cache the prediction. Fifth — monitoring: Prometheus metrics on request rate, latency p95/p99, cache hit rate, error rate; auto-scale when p95 > 50ms."

---

**A7. How would you move this entire project to Databricks?**

Testing: Platform migration thinking.

Strong answer: "Component-by-component: Data ingestion → Databricks Auto Loader or notebooks reading from Delta Lake tables. Preprocessing → PySpark UDFs for distributed text cleaning; large-scale stratified splits using Spark's randomSplit with stratification. Training → MLflow is native in Databricks (same API), but training would use a Databricks Job cluster. At scale, use Spark ML or distribute scikit-learn training across nodes with joblib + Spark's barrier mode. MLflow Registry → Unity Catalog Model Registry (replaces local MLflow registry, adds governance, lineage). Model serving → Databricks Model Serving (managed endpoint, auto-scales, supports A/B testing). Monitoring → Databricks Lakehouse Monitoring on Delta tables. CI/CD → Databricks Asset Bundles (DABs) for deploying Jobs; GitHub Actions triggers Databricks REST API to run jobs."

---

**A8. What are the production risks of the current model promotion strategy?**

Testing: Production MLOps risk awareness.

Strong answer: "Five risks: 1) No rollback automation — if the promoted model degrades in production, rollback is manual (MLflow UI → transition previous version). 2) Accuracy as the only gate — the model could pass accuracy threshold but have extremely poor recall on one class that matters to the business. 3) No canary deployment — 100% of traffic immediately sees the new model; a bad model affects all users at once. 4) Test set contamination risk — if the test.csv was ever inadvertently used for model selection or tuning, the threshold check is optimistic. 5) Model staleness — promotion is one-directional; there's no automated mechanism to detect that a Production model has degraded over time and needs replacement."

---

**A9. How would you reproduce an exact historical model training run?**

Testing: Reproducibility in practice.

Strong answer: "Four steps: 1) Find the MLflow run: use the MLflow UI or client to find the run_id with the target model. 2) Get the exact code: from MLflow run tags, retrieve mlflow.source.git.commit (the git commit hash). Run `git checkout <commit_hash>` to restore exact source code. 3) Get the exact config: the params are logged in MLflow — but also check out the params.yaml from that git commit. 4) Get the exact data: if DVC is configured with a remote, run `dvc checkout` to restore the exact data version tracked at that commit. Then run `python src/train.py`. The output should be bit-for-bit identical because random_state is fixed. Limitation: this project doesn't have a DVC remote, so you'd need to re-ingest data — which could differ if HuggingFace's dataset has been updated."

---

**A10. Design an automated retraining pipeline for this project.**

Testing: Advanced MLOps architecture.

Strong answer: "Trigger → Evaluation → Decision → Retrain → Validate → Deploy:

TRIGGER: Scheduled (weekly) OR drift alert (PSI > 0.15) OR performance degradation (new labeled sample accuracy drops > 5% from baseline).

EVALUATION: Before retraining, quantify: fetch 500 recent predictions, have them labeled by human annotators or a proxy labeler, compute accuracy on this recent sample. Compare to baseline.

DECISION GATE: If recent accuracy >= 0.88 AND drift < 0.15, no retraining needed. Else proceed.

RETRAINING: GitHub Actions workflow_dispatch triggers with `force_train: true`. Pipeline runs with expanded data: original training data + recent production data (collected from API logs). New model trained and evaluated.

VALIDATION: New model must beat the current Production model on the validation set AND the recent labeled sample AND meet the accuracy threshold.

DEPLOYMENT: If validation passes, register new version. Deploy to Staging. Run automated integration tests. If tests pass, promote to Production. Previous version → Archived.

MONITORING: Track rollout — are error rates stable? Are prediction distributions stable post-rollout? Auto-rollback if p95 latency > SLA or error rate > 0.1%.

This project implements the retraining STEP but not the automated TRIGGER or DECISION logic."
