# TOPIC 11: DVC & Data Versioning

---

## 1. What is DVC?

DVC (Data Version Control) is an open-source tool that extends Git to handle large files, datasets, and ML pipelines. It adds two things Git cannot do:

1. **Data versioning**: Store large files (CSVs, models) in remote storage (S3, GCS) and track them in Git using small pointer files
2. **Pipeline automation**: Define a DAG (Directed Acyclic Graph) of pipeline stages with dependency tracking and caching

DVC commands mirror Git commands intentionally: `dvc add`, `dvc push`, `dvc pull`, `dvc repro`.

---

## 2. Why Git Alone Is Insufficient for ML

| Problem | Git's limitation |
|---------|-----------------|
| `data/raw/train_raw.csv` is 50MB | Git max file recommendation is ~50MB; GitHub hard limit is 100MB |
| Full AG News dataset is ~500MB | Impossible to commit to GitHub |
| Model .pkl files are 200MB+ | Same problem |
| Git history grows forever | 10 training runs × 200MB = 2GB of history |
| Slow collaboration | Everyone waits to clone a 10GB repo |

**DVC's solution:** Store the data in remote storage (S3). Store a small `.dvc` pointer file in Git. When a team member checks out the code, they run `dvc pull` to download the actual data.

```
Git stores:           data/raw/train_raw.csv.dvc  (50 bytes — the pointer)
S3/GCS stores:        data/raw/train_raw.csv  (50 MB — the actual data)
```

---

## 3. DVC in THIS Repository

**File:** `dvc.yaml`

This project uses DVC for **pipeline automation** (defining stages and their dependencies), not for remote data storage (no DVC remote is configured).

```yaml
# dvc.yaml
stages:
  ingest:
    cmd: python src/ingest.py          # command to run
    deps:
      - src/ingest.py                  # if THIS changes → re-run
      - params.yaml                    # if THIS changes → re-run
    params:
      - data.dataset                   # if THESE params change → re-run
      - data.max_samples
    outs:
      - data/raw/train_raw.csv         # output files (DVC tracks these)
      - data/raw/test_raw.csv

  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/train_raw.csv         # depends on ingest output
      - data/raw/test_raw.csv
      - params.yaml
    params:
      - data.test_size
      - data.val_size
      - data.random_state
    outs:
      - data/processed/train.csv
      - data/processed/val.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv       # depends on preprocess output
      - data/processed/val.csv
      - params.yaml
    params:
      - model
      - training
      - mlflow.experiment_name
    outs:
      - models/

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/
      - data/processed/test.csv
      - params.yaml
    metrics:
      - reports/final_metrics.json:
          cache: false                  # don't cache this → always fresh
    plots:
      - reports/final_confusion_matrix.png:
          cache: false

  monitor:
    cmd: python monitoring/monitor.py
    deps:
      - monitoring/monitor.py
      - data/processed/train.csv
      - data/processed/test.csv
      - params.yaml
    outs:
      - reports/drift_report.html:
          cache: false
```

---

## 4. DVC Pipeline Execution

### Running the full pipeline:
```bash
dvc repro
```

DVC checks each stage:
1. Are the `deps` (source files + input data) unchanged?
2. Are the `params` unchanged?
3. Are the `outs` (outputs) already present and match the stored hash?

If ALL three match → **cache hit** → skip this stage.
If ANY changed → **re-run this stage** AND all downstream stages.

### Example: Change max_samples in params.yaml

```
params.yaml: max_samples: 10000 → 20000

DVC analysis:
  ingest.py:        params changed (data.max_samples) → RE-RUN
  preprocess.py:    deps changed (train_raw.csv is new) → RE-RUN
  train.py:         deps changed (train.csv, val.csv are new) → RE-RUN
  evaluate.py:      deps changed (models/ is new) → RE-RUN
  monitor.py:       deps changed (train.csv is new) → RE-RUN
  
All stages re-run. Makes sense — changing sample size invalidates everything.
```

### Example: Fix a bug in src/serve.py

```
DVC analysis:
  ingest.py:     serve.py is not a dep → no change
  preprocess.py: serve.py is not a dep → no change
  train.py:      serve.py is not a dep → no change
  evaluate.py:   serve.py is not a dep → no change
  monitor.py:    serve.py is not a dep → no change

No stages re-run. Correct — serve.py changes don't affect model training.
```

---

## 5. DVC .dvc Files (Data Versioning)

When you run `dvc add data/raw/train_raw.csv`, DVC:
1. Moves the file to a content-addressable cache (`.dvc/cache/`)
2. Creates `data/raw/train_raw.csv.dvc`:
```yaml
outs:
- md5: a1b2c3d4e5f6...   # MD5 hash of the file contents
  size: 52428800           # file size in bytes
  path: train_raw.csv
```
3. Adds `data/raw/train_raw.csv` to `.gitignore`

You commit the `.dvc` file to Git. The actual data goes to remote storage.

**This project does NOT have a DVC remote configured.** The pipeline stages are defined in `dvc.yaml`, but there's no `dvc push/pull` for remote storage — data files are generated locally.

---

## 6. The Dependency Graph (DAG)

```
params.yaml ──────────────────────────────────┐
    │                                          │
    ▼                                          │
src/ingest.py ──────► data/raw/               │
                          │                    │
                          ▼                    │
                  src/preprocess.py ──────────►│
                          │                    │
                          ▼                    │
                  data/processed/              │
                     │        │               │
                     ▼        │               │
              src/train.py ◄──┘               │
                     │                        ▼
                     ▼                 src/evaluate.py
                  models/ ────────────────────►│
                                               ▼
                                      reports/final_metrics.json
                                      reports/final_confusion_matrix.png
```

DVC computes this graph automatically from the `deps`/`outs` declarations. It only re-runs what's necessary.

---

## 7. DVC vs Git for ML

| Concern | Git | DVC |
|---------|-----|-----|
| Code versioning | ✅ Native | Uses Git for code |
| Large file tracking | ❌ Breaks | ✅ Pointer files |
| Remote data storage | ❌ GitHub has file limits | ✅ S3, GCS, Azure Blob |
| Pipeline DAG | ❌ No concept | ✅ dvc.yaml |
| Selective re-run | ❌ You must manage | ✅ Automatic cache invalidation |
| Data version history | ❌ Can't version large files | ✅ Each commit can have a different dataset version |
| Team collaboration | ✅ Pull requests | ✅ `dvc push/pull` to share data |

---

## 8. Reproducibility with DVC

Given a Git commit + DVC, you can reproduce any historical experiment:

```bash
git checkout <commit_hash>   # restore code + params
dvc checkout                 # restore exact data files for that commit
dvc repro                    # re-run the pipeline
```

This gives you the SAME data, SAME code, SAME parameters → SAME model.

---

## 9. Practical DVC Commands

```bash
# Initialize DVC in a repo
dvc init

# Add a data file to DVC tracking
dvc add data/raw/train_raw.csv

# Configure S3 remote storage
dvc remote add -d myremote s3://my-bucket/mlops

# Push data to remote
dvc push

# Pull data from remote (e.g., on a new machine)
dvc pull

# Run the pipeline (re-runs only changed stages)
dvc repro

# Check pipeline status (what would be re-run?)
dvc status

# Compare metrics between experiments
dvc metrics show
dvc metrics diff

# Visualize the pipeline DAG
dvc dag
```

---

## 10. Production Perspective

**What this repo implements:**
- DVC pipeline definition (`dvc.yaml`) with dependency tracking
- Automatic cache invalidation when deps/params change
- Metrics and plots tracking in evaluate stage

**What a production system would add:**
- **DVC remote** (S3/GCS): Push training data so all team members can pull it
- **DVC experiments** (`dvc exp run`): Track experiment branches similarly to Git branches
- **CML (Continuous Machine Learning)**: DVC's CI/CD integration that posts metrics to pull requests
- **Data versioning**: Lock down the exact dataset version used for production training

---

## 11. Interview Questions

**Q1 (Beginner): Why can't you just use Git to track your training data?**
- Strong: "Git is designed for text files and small binaries. Large files (50MB+ CSVs, 200MB model files) cause Git to become extremely slow, blow up repository size, and exceed GitHub's file size limits. DVC solves this by storing only a small pointer file in Git and pushing the actual data to cloud storage (S3, GCS). This way, Git stays fast and your data is properly versioned."

**Q2 (Intermediate): Explain how `dvc repro` knows which stages to re-run.**
- Strong: "DVC hashes all dependencies for each stage: the source files, input data files, and parameter values. Between runs, it compares these hashes to what's stored in the DVC cache. If any dep hash changed, the stage is marked as stale and must re-run. Downstream stages that depend on stale outputs also re-run. This is the same principle as Make targets, but applied to ML pipelines."

**Q3 (Advanced): This project defines dvc.yaml but has no DVC remote configured. What is the practical limitation?**
- Strong: "Without a remote, DVC only provides local pipeline automation — it can avoid re-running stages, but data files can't be shared between machines. If a new developer joins or CI/CD needs the data, they must run the full ingestion pipeline themselves. In a team environment, you'd configure `dvc remote add -d s3://bucket` and run `dvc push` after generating data. Teammates run `dvc pull` to get the exact same data without re-downloading from HuggingFace."

---

## Must Know
- DVC tracks large files with small pointer (.dvc) files in Git
- `dvc repro` re-runs only changed stages based on dependency hashing
- `dvc.yaml` defines stages, deps, params, and outs
- A DVC remote (S3/GCS) is needed for team data sharing
- DVC enables full experiment reproducibility: code (Git) + data (DVC)

## Should Know
- `dvc add`, `dvc push`, `dvc pull` workflow
- How to read dvc.yaml stages
- The DAG structure and how downstream invalidation works
- `metrics:` vs `outs:` in dvc.yaml (metrics are trackable, comparable between experiments)
- `cache: false` means the output is tracked but not cached (always considered fresh)

## Nice to Know
- `dvc exp run` for experiment branches
- CML (Continuous Machine Learning) for CI/CD integration
- DVC Studio (SaaS dashboard for DVC experiments)
- MLflow + DVC integration patterns
