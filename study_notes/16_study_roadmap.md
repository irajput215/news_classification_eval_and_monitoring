# TOPIC 16: Personalized Study Roadmap

---

## What You Already Know

| Skill | Proficiency | Implication |
|-------|-------------|-------------|
| Python | Strong | No basics needed |
| SQL | Strong | Skip all SQL tutorials |
| PySpark | Strong | MLlib, distributed training concepts will be fast |
| Databricks | Strong | Platform knowledge transfers directly |
| Basic ML | Good | Algorithm internals already covered |
| PyTorch | Good | Training loops, loss functions understood |
| FastAPI | Good | serve.py is immediately readable |
| Docker | Good | Dockerfile concepts are familiar |
| RAG/GenAI | Good | LLM patterns, embedding concepts understood |

---

## What You Need to Build

| Gap | Priority | Why |
|-----|----------|-----|
| MLflow (tracking + registry) | **CRITICAL** | Asked in nearly every ML Engineer interview |
| Model Registry concepts | **CRITICAL** | The lifecycle management story |
| Evaluation metrics deeply | **HIGH** | F1, AUC-ROC, PR-AUC must be intuitive, not memorized |
| Data leakage prevention | **HIGH** | Classic ML interview trap question |
| ML Pipelines (sklearn) | **HIGH** | Training-serving consistency |
| Model monitoring | **HIGH** | Increasingly asked at senior ML Engineer level |
| Data/feature drift | **HIGH** | Core to production ML |
| Retraining decisions | **MEDIUM** | Senior-level concept |
| Hyperparameter tuning | **MEDIUM** | Optuna, cross-validation |
| MLOps architecture | **MEDIUM** | Big-picture design questions |
| Concept drift | **MEDIUM** | Hard to implement, but asked conceptually |
| Time-series specifics | **LOW** | Not in this project; learn after other gaps |

---

## Recommended Study Sequence

### Phase 1: Foundation (Week 1) — Do This First

**Start here because everything else depends on it.**

1. **Read File 06** (Model Evaluation) — You MUST internalize accuracy vs F1 vs AUC-ROC. Be able to explain each from first principles without notes.

2. **Understand data leakage** — Why does splitting AFTER preprocessing cause leakage? Memorize: split first, then fit preprocessing on train only, transform val/test.

3. **Read File 04** (Training Pipelines) — sklearn Pipeline is the central mechanism. Understand why it prevents leakage.

4. **Do**: Open train.py, trace exactly what happens inside `run_experiment()`. Map each line to an MLflow concept.

---

### Phase 2: MLflow Deep Dive (Week 1-2) — Most Important

**MLflow will be asked in essentially every ML Engineer interview.**

1. **Read File 05 completely** — Do not skip any section.

2. **Practice run**: Set up a local MLflow server (`mlflow server`), write a 20-line script that trains a LogisticRegression, logs params/metrics/model, registers it.

3. **Key concepts to memorize:**
   - Experiment → Run → Params/Metrics/Artifacts → Model → Registry → Version → Stage
   - The `runs:/` and `models:/` URI formats
   - What `infer_signature()` does and why
   - The context manager pattern `with mlflow.start_run()`

4. **Do**: Look at the MLflow UI for this project. Find the 3 training runs. Compare their val_accuracy. Find the registered model. See its version and stage.

---

### Phase 3: Production Concepts (Week 2) — High Priority

1. **Read File 07** (Deployment) — Understand the FastAPI architecture. Know all endpoint types.

2. **Read File 10** (Drift & Monitoring) — PSI formula, when to retrain decision framework.

3. **Read File 03** (Preprocessing) — CRITICAL: training-serving skew. This is a classic interview trap.

4. **Key concepts to memorize:**
   - Training-serving skew: what it is, why it happens, how to fix it
   - PSI thresholds: < 0.1 stable, 0.1-0.2 monitor, > 0.2 investigate
   - Retraining decision: drift alone is NOT enough; need performance degradation evidence

---

### Phase 4: MLOps Architecture (Week 3)

1. **Read File 09** (CI/CD) — Understand the 4-job structure. Know what each job does and why.

2. **Read File 11** (DVC) — Understand `dvc repro` and cache invalidation.

3. **Read File 13** (End-to-End Walkthrough) — Practice this OUT LOUD. Time yourself. Aim for 3 minutes.

4. **Read File 14** (Interview Questions) — Answer each question without looking at the answer first.

---

### Phase 5: Databricks Integration (Week 3-4)

1. **Read File 15** (Databricks Mapping) — Map every concept in this project to Databricks equivalents.

2. Since you already know Databricks, focus on:
   - Unity Catalog Model Registry (vs local MLflow Registry)
   - Databricks Model Serving (vs FastAPI Docker container)
   - Databricks Lakehouse Monitoring (vs manual PSI scripts)

---

## Dependency Graph

```
ML Fundamentals (You have this)
        ↓
Evaluation Metrics (File 06)
        ↓
Data Leakage (File 03, training-serving section)
        ↓
sklearn Pipeline (File 04)
        ↓
MLflow Tracking (File 05 — sections 1-8)
        ↓
MLflow Model Registry (File 05 — sections 9-14)
        ↓
Model Deployment (File 07)
        ↓
Containerization (File 08)
        ↓
CI/CD (File 09)
        ↓
Monitoring & Drift (File 10)
        ↓
Data Versioning (File 11)
        ↓
Retraining Decisions (File 10, section 11 + File 06, section 8)
        ↓
MLOps Architecture (File 00 README + File 13)
        ↓
Databricks Migration (File 15)
```

---

## For ML Engineer Interviews — Must Know

These are non-negotiable for ML Engineer roles:

1. **MLflow end-to-end**: Can you explain experiments → runs → params/metrics → model logging → registry → stages → serving?

2. **Training-serving skew**: What is it? How does this project have it? How do you fix it?

3. **Data leakage prevention**: Fit preprocessing on train only. sklearn Pipeline enforces this.

4. **Model evaluation**: Accuracy vs F1 vs AUC-ROC. When to use each. Multi-class averaging.

5. **Model versioning and rollback**: How do you roll back? Model Registry version management.

6. **Health checks**: Why do serving APIs need them? How do load balancers use them?

7. **CI/CD quality gate**: What prevents a bad model from reaching production in this project?

8. **Data drift**: Feature drift vs concept drift. PSI as a detection method.

9. **When to retrain**: PSI alone is not sufficient. Need performance degradation evidence.

10. **Docker multi-stage**: Why two stages? What does each do?

---

## For Data Engineer Interviews — Must Know

DE interviews focus on data quality and pipeline reliability:

1. **Data validation**: What checks does ingest.py perform? What are hard vs soft failures?

2. **DVC pipeline**: How does `dvc repro` know what to re-run? What triggers cache invalidation?

3. **Stratified splitting**: Why stratify? What goes wrong without it?

4. **Schema enforcement**: REQUIRED_COLUMNS check. Why it's a hard failure.

5. **Monitoring**: What 6 features are monitored? Why proxy features instead of raw text?

6. **Docker Compose networking**: How do containers communicate? Service DNS resolution.

7. **Data versioning**: Why can't Git version large files? What does DVC store in Git?

---

## For AI Engineer Interviews — Also Useful

AI engineers increasingly need MLOps foundations:

1. **MLflow for LLM evaluation**: Same tracking API applies to prompt engineering experiments

2. **Model serving patterns**: FastAPI patterns transfer to LLM serving (OpenAI-compatible APIs)

3. **Drift monitoring for LLMs**: Semantic drift in embeddings — harder than feature drift, same PSI approach

4. **Retrieval-Augmented Generation**: Data ingestion and validation patterns from this project apply to RAG pipelines

5. **Model registry for LLMs**: Registering fine-tuned models, managing prompt versions

---

## Topics You Can Skip Initially

1. **Time-series specifics** — Not in this project. Learn only if you interview for time-series roles.

2. **PyTorch training loops** — You already know this; this project uses scikit-learn.

3. **Advanced hyperparameter search** (Optuna, Ray Tune) — Know conceptually, but not needed for basic ML Engineer interviews.

4. **ONNX / model format conversion** — Edge case; skip until you encounter it.

5. **Kubernetes details** — Know concepts (replicas, load balancing, probes), not YAML syntax.

---

## 30-Day Study Plan

### Week 1: Core ML + MLflow
- Day 1-2: File 06 (Evaluation) — master ALL metrics
- Day 3: File 03 (Preprocessing) — training-serving skew
- Day 4: File 04 (Training Pipelines) — sklearn Pipeline deeply
- Day 5-7: File 05 (MLflow) — run the actual code, use the UI

### Week 2: Production Systems
- Day 8-9: File 07 (FastAPI Deployment)
- Day 10: File 08 (Docker)
- Day 11-12: File 09 (CI/CD)
- Day 13-14: File 10 (Drift Monitoring)

### Week 3: Architecture + Interviews
- Day 15: File 11 (DVC)
- Day 16: File 12 (Testing)
- Day 17-18: File 13 (End-to-End Walkthrough) — practice out loud
- Day 19-21: File 14 (Interview Questions) — answer without notes

### Week 4: Databricks + Deep Dives
- Day 22-23: File 15 (Databricks Mapping)
- Day 24-25: Revisit weak areas from mock interviews
- Day 26-30: Mock interviews, system design practice

---

## Must Know (Absolute Priority)

1. MLflow experiment → run → params/metrics/artifacts → model → registry → stage
2. Training-serving skew: what, why, fix
3. Data leakage: fit on train only, transform val/test
4. F1, AUC-ROC, Precision, Recall — explain from first principles
5. PSI < 0.1 stable, 0.1-0.2 monitor, > 0.2 investigate
6. Retraining = drift + performance degradation (not drift alone)
7. Model rollback = promote previous version back to Production
8. Health check = load balancer signals service availability
9. Multi-stage Docker = lean production image (no build tools)
10. CI quality gate = non-zero exit code blocks downstream jobs

## Should Know (High Priority)

1. DVC repro cache invalidation mechanism
2. Stratified split math (relative_val calculation)
3. CalibratedClassifierCV — why SVMs need it
4. `depends_on: condition: service_healthy` in Docker Compose
5. `$GITHUB_OUTPUT` for passing values between CI steps
6. Evidently fallback to PSI — graceful degradation pattern
7. `scope="module"` pytest fixtures
8. Pydantic validators (`@field_validator`)
9. MLflow model signature — input/output type contract
10. Databricks equivalents for each this-project component

## Nice to Know (After You Have the Above)

1. Optuna/Ray Tune for automated hyperparameter search
2. Platt scaling details (how CalibratedClassifierCV works internally)
3. KL divergence vs Jensen-Shannon divergence vs Wasserstein
4. ONNX model format
5. Triton Inference Server for GPU serving
6. MLflow aliases vs stages (UC new vs old API)
7. ADWIN/DDM streaming drift algorithms
8. CML (Continuous Machine Learning) DVC integration
9. Databricks Asset Bundles (DABs) syntax
10. Model card generation
