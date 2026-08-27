# TOPIC 01: ML Project Structure & Configuration Management

---

## 1. What is it?

**ML Project Structure** is a deliberate organization of files and directories so that every component of the ML lifecycle has a predictable location. Without a standard structure, notebooks sprawl, paths get hardcoded, and the project becomes unmaintainable.

**Configuration Management** means keeping all tunable values (hyperparameters, paths, thresholds, URIs) in one file, separate from code, so that:
- Changing a value does NOT require touching source code
- The config file becomes a record of exactly what was used for any given experiment
- CI/CD can override values without code changes

---

## 2. Why do we need it?

**Without it:**
- You have `train_v2_final_FINAL.py`, `train_v3_actually_final.py`
- Paths like `C:/Users/john/Documents/data/train.csv` are hardcoded
- You have no idea which hyperparameters produced which model
- Adding a new team member requires 2 hours of "setup by tribal knowledge"

**With it:**
- Any developer can `git clone` → run pipeline immediately
- One change in `params.yaml` changes the experiment
- Every run is traceable back to the config used

---

## 3. Where is it used in THIS repository?

### Project Structure:
```
mlops-pipeline/
├── src/                  ← All pipeline stage scripts
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve.py
├── monitoring/           ← Drift monitoring (separate from src/ — important!)
│   └── monitor.py
├── tests/                ← Pytest test suite
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .github/workflows/    ← CI/CD automation
│   └── ci.yml
├── data/                 ← NOT committed (generated at runtime)
│   ├── raw/              ← Output of ingest.py
│   └── processed/        ← Output of preprocess.py
├── models/               ← Output of train.py (pkl files)
├── reports/              ← Output of evaluate.py + monitor.py
├── mlruns/               ← MLflow local artifact store
├── params.yaml           ← CENTRAL CONFIG ← most important file
├── dvc.yaml              ← Pipeline orchestration
├── Dockerfile            ← Container definition
├── docker-compose.yml    ← Multi-service orchestration
└── requirements.txt      ← Python dependencies
```

### Configuration File:
**File:** `params.yaml`

```yaml
data:
  dataset: "ag_news"          # or "truthlens"
  test_size: 0.15
  val_size: 0.15
  random_state: 42
  max_samples: 10000

model:
  type: "tfidf_lr"
  max_features: 50000
  ngram_range: [1, 2]

training:
  learning_rate: 0.01
  C: 1.0
  max_iter: 1000
  class_weight: "balanced"

mlflow:
  experiment_name: "news-classification-mlops"
  tracking_uri: "http://mlflow:5000"
  registered_model_name: "news_classifier"
  accuracy_threshold: 0.88

serving:
  host: "0.0.0.0"
  port: 8000
  model_stage: "Production"

monitoring:
  drift_threshold: 0.15
  reference_data: "data/processed/train.csv"
  report_path: "reports/drift_report.html"
```

---

## 4. How does it work in THIS project?

Every script reads `params.yaml` at startup using the SAME pattern:

```python
ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
```

**Line by line:**
- `Path(__file__).resolve().parent.parent` → Navigate 2 levels up from any script to find the repo root. This means scripts work regardless of where you run them from.
- `yaml.safe_load(open(...))` → Parse YAML into a Python dict. `safe_load` prevents executing arbitrary Python (security best practice vs `yaml.load`).

Then each script accesses what it needs:
```python
# ingest.py
dataset     = PARAMS["data"]["dataset"]        # "ag_news"
max_samples = PARAMS["data"]["max_samples"]    # 10000

# train.py
tracking_uri = PARAMS["mlflow"]["tracking_uri"]
threshold    = PARAMS["mlflow"]["accuracy_threshold"]

# monitor.py
drift_threshold = PARAMS["monitoring"]["drift_threshold"]
```

---

## 5. Code Walkthrough

```python
# Pattern used in EVERY script (src/train.py line 43-44)
ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
```

| Line | What it does | Why it exists | What breaks without it |
|------|-------------|---------------|----------------------|
| `Path(__file__).resolve()` | Gets absolute path of the current file | Makes path resolution OS-independent | Paths break when running from different directories |
| `.parent.parent` | Goes up 2 dirs (src/ → repo root) | All scripts are 1 level deep in src/ | Would need hardcoded absolute paths |
| `yaml.safe_load(...)` | Parses YAML → Python dict | Config as code; no hardcoding | You'd need to change source code for every experiment |

---

## 6. Input → Processing → Output

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| Config load | `params.yaml` (YAML text) | `yaml.safe_load()` | Python dict `PARAMS` |
| Script startup | `PARAMS` dict | Index by key | Hyperparams ready for use |
| DVC stage change | Edit `params.yaml` | `dvc repro` detects change | Only affected stages re-run |

---

## 7. Production Perspective

**What this repository does:**
- Single YAML config file, read at script startup
- All scripts share the same config format

**What a production system would add:**
- Environment-specific configs (dev/staging/prod)
- Secrets management (AWS Secrets Manager, Vault) — NOT in YAML
- Config validation with Pydantic or similar
- Config versioning alongside code

**What could fail:**
- Someone commits a config with wrong `tracking_uri` → model registers to wrong experiment
- `max_samples: null` (full dataset) accidentally committed → CI times out

**How to detect:** Compare config hash between environments. Alert on config changes in CI.

---

## 8. Important MLOps Concepts

**Implemented in this repo:**
- ✅ **Separation of config from code** — `params.yaml` vs `src/*.py`
- ✅ **Reproducibility** — `random_state: 42` everywhere in config
- ✅ **Single source of truth** — one file controls all stages

**Not implemented but important:**
- ❌ **Environment-based config override** — no dev/prod split
- ❌ **Secret management** — tracking URI contains no credentials here
- ❌ **Config validation** — no schema check on `params.yaml`

---

## 9. Common Mistakes

1. **Hardcoding paths** — `pd.read_csv("/Users/john/data.csv")` → breaks on every other machine
2. **Storing secrets in config** — Never put API keys in `params.yaml` (use env variables)
3. **Not versioning config** — Changing `params.yaml` without a Git commit means you can't reproduce that experiment
4. **Config in multiple places** — Same threshold defined in both `params.yaml` and hardcoded in a script
5. **Using `yaml.load()` not `yaml.safe_load()`** — Security vulnerability

---

## 10. Interview Questions

**Q1 (Beginner): Why is configuration management important in MLOps?**
- Testing: Do you understand separation of concerns?
- Strong answer: "Without centralized config, you'd need to change source code to change hyperparameters. That means every experiment could have code changes mixed with parameter changes, making it impossible to reproduce results. Config files keep experiments deterministic and reproducible."
- Weak answer: "It makes the project cleaner."
- Follow-up: "How would you handle configs that differ between dev and production?"

**Q2 (Intermediate): In this project, why does every script compute ROOT using `Path(__file__).resolve().parent.parent` instead of using `os.getcwd()`?**
- Testing: Do you understand path resolution?
- Strong answer: "`os.getcwd()` returns the directory where the process was LAUNCHED from, which changes depending on how you invoke the script. `Path(__file__)` always points to the script's own location, regardless of the working directory. This makes the scripts portable."
- Follow-up: "What happens if you run `python src/train.py` from the repo root vs from inside `src/`?"

**Q3 (Advanced): How would you implement environment-specific configuration (dev/staging/prod) for this project?**
- Testing: Production system design knowledge
- Strong answer: "I'd use a base config (`params.yaml`) and environment-specific overrides (`params.prod.yaml`, `params.dev.yaml`). At startup, merge them. For secrets like DB credentials or API keys, use environment variables or a secrets manager — never YAML. In CI/CD, the environment name determines which config is used."
- Follow-up: "How would you prevent a developer accidentally running training with the production MLflow URI?"

---

## 11. Connection to Other MLOps Concepts

```
params.yaml
    ↓ controls
Data Ingestion (dataset choice, max_samples)
    ↓
Preprocessing (test_size, val_size, random_state)
    ↓
Training (model type, C, max_features, ngram_range)
    ↓
MLflow (tracking_uri, experiment_name, accuracy_threshold)
    ↓
Deployment (model_stage, host, port)
    ↓
Monitoring (drift_threshold, reference_data path)
    ↓
DVC (tracks which params each stage depends on → cache invalidation)
    ↓
CI/CD (patches params.yaml to point to CI MLflow URI, not Docker network)
```

---

## Must Know
- Every ML project needs a root config file (YAML, TOML, or JSON)
- Never hardcode paths or hyperparameters in source code
- `random_state=42` everywhere makes results reproducible
- Config should be versioned in Git alongside the code
- Secrets go in environment variables, not config files
- `yaml.safe_load()` not `yaml.load()`

## Should Know
- Environment-specific config override patterns
- Pydantic for config validation
- How DVC uses params.yaml for cache invalidation
- Config merging strategies

## Nice to Know
- Hydra (Facebook) for hierarchical config management
- MLflow `log_param()` can also serve as config versioning
- HashiCorp Vault for secrets management

---

## Study Checkpoint Quiz

**Conceptual:**
1. Why does separating config from code improve reproducibility?
2. What is the difference between `yaml.safe_load()` and `yaml.load()`?
3. Why should secrets never be stored in `params.yaml`?

**Scenario:**
4. You need to run training with 50,000 samples instead of 10,000. Where exactly do you make the change, and what is the risk if you hardcode it instead?
5. You're onboarding a new team member. What's the minimal set of files they need to understand to reproduce your results?

**Repository-specific:**
6. Why does `ci.yml` patch `params.yaml` before running training in GitHub Actions?

**Interview:**
7. The interviewer asks: "How do you ensure that the model in production was trained with the same parameters as what's documented?" Give a strong answer.

---

## Answers

1. Config separation means the code logic is identical across runs. Only the config changes. If you version your config in Git (or log it to MLflow), you can always reproduce any experiment by checking out that config.
2. `safe_load` only handles basic YAML data types (strings, numbers, lists, dicts). `yaml.load` can execute arbitrary Python constructors — a security risk if loading untrusted YAML.
3. Secrets in config files get committed to Git, accidentally shared, or exposed in CI logs.
4. Change `max_samples: 50000` in `params.yaml`. If hardcoded, you create a code change that looks like a feature change in Git history, making it impossible to trace why results changed.
5. `params.yaml`, `requirements.txt`, `dvc.yaml`, and the `src/` directory.
6. In CI, there's no Docker network, so `http://mlflow:5000` (the Docker service hostname) doesn't resolve. The CI MLflow server runs on `localhost:5000`, so params.yaml must be patched.
7. "MLflow logs all parameters at the start of every run. I can go to the MLflow UI, find the run that produced the registered model, and see the exact params dict. Additionally, params.yaml is version-controlled, so I can tie the Git commit hash logged in MLflow back to the exact config."
