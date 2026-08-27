# TOPIC 09: CI/CD & GitHub Actions

---

## 1. What is CI/CD?

**CI (Continuous Integration):** Every code change is automatically tested. The moment you push code, an automated system runs your tests. If tests fail, you know immediately — before merging.

**CD (Continuous Delivery/Deployment):** After CI passes, code is automatically packaged and deployed. Two variants:
- **Continuous Delivery**: Automatically prepares release package (manual deploy trigger)
- **Continuous Deployment**: Automatically deploys to production (no manual step)

**In MLOps, CI/CD extends to:**
- Testing data quality (not just code)
- Training models (not just building software)
- Evaluating model performance (not just unit tests)
- Building and pushing Docker images
- Running drift detection

---

## 2. Why CI/CD in MLOps?

Without CI/CD:
```
Developer changes preprocessing → tests locally → looks fine
PR merged → training runs → model degrades → deployed to production
Nobody knew because nothing was automated
```

With CI/CD:
```
Developer changes preprocessing → git push
→ Automated: lint, test_data, test_model, test_api (tests run in <5 min)
→ If any test fails: PR blocked, developer notified immediately
→ If all pass: training runs, model evaluated, if accuracy meets threshold → Docker image built and pushed
```

---

## 3. Where is it used in THIS repository?

**File:** `.github/workflows/ci.yml`

**Trigger:** Every push to `main` or `dev` branches (paths filtered — only if src/, tests/, monitoring/, params.yaml, requirements.txt, or Dockerfile changes)

**Also triggered by:** Pull requests to `main`, and manual dispatch (`workflow_dispatch`)

---

## 4. The 4-Job Pipeline — Complete Walkthrough

```
┌─────────────────────────────────────────┐
│  Job 1: test                            │
│  🧪 Lint & Unit Tests (always runs)     │
│  → Run: ubuntu-latest                   │
│  → Timeout: 20 min                      │
└────────────────┬────────────────────────┘
                 │ needs: test (must pass)
                 ▼
┌─────────────────────────────────────────┐
│  Job 2: train-and-evaluate              │
│  🏋️ Train & Evaluate                   │
│  → Only on main branch, not PRs        │
│  → Timeout: 60 min                      │
└────────────────┬────────────────────────┘
                 │ needs: train-and-evaluate (must pass)
                 ▼
┌─────────────────────────────────────────┐
│  Job 3: docker-build                    │
│  🐳 Build & Push Docker Image           │
│  → Only on main branch, not PRs        │
│  → Requires: DOCKERHUB_USERNAME + TOKEN │
└────────────────┬────────────────────────┘
                 │ needs: all 3 above
                 ▼
┌─────────────────────────────────────────┐
│  Job 4: notify                          │
│  📢 Pipeline Summary                    │
│  → Always runs (even if others fail)   │
│  → Writes summary to GitHub UI         │
└─────────────────────────────────────────┘
```

---

## 5. Job 1: Test (Line-by-Line)

```yaml
test:
  name: 🧪 Lint & Test
  runs-on: ubuntu-latest     # fresh Ubuntu VM for every run
  timeout-minutes: 20

  steps:
    - name: Checkout code
      uses: actions/checkout@v4     # clones the repo

    - name: Set up Python 3.10
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
        cache: pip                  # cache pip downloads for speed

    - name: Install dependencies
      run: |
        pip install scikit-learn numpy pandas fastapi uvicorn pydantic \
          mlflow joblib pyyaml matplotlib seaborn pytest pytest-asyncio httpx evidently

    - name: Generate synthetic CI data   # ← KEY INSIGHT
      run: |
        python - <<'EOF'
        # Creates fake data/processed/{train,val,test}.csv
        # WITHOUT downloading from HuggingFace (would take too long)
        texts = ["Stock market...", "Football..."] * 300
        ...
        EOF
```

**Why synthetic CI data?** Downloading AG News from HuggingFace takes 2+ minutes and requires internet. CI must be fast (<5 min) and reliable. The synthetic data has the same schema — tests validate schema and API behavior, not model performance.

```yaml
    - name: Run data tests
      run: |
        pytest tests/test_data.py -v --tb=short \
          --deselect tests/test_data.py::TestSplitIntegrity::test_no_overlap_train_test
```

**Why `--deselect`?** The overlap test fails with synthetic data (same 4 texts repeated → train and test have identical texts). This is a known CI limitation, not a real bug. The test is deselected in CI but runs in full training.

```yaml
    - name: Train quick CI model
      run: |
        python - <<'EOF'
        # Train a tiny 5000-feature model on synthetic data
        # Saves as models/tfidf_svm_bigrams.pkl  ← test_model.py expects this
        EOF
```

**Why train in CI?** `test_model.py` loads a `.pkl` file and tests its output shape and prediction values. CI needs a real model, but doesn't need the production model.

```yaml
    - name: Run API tests
      run: pytest tests/test_api.py -v --tb=short
```

API tests mock the model (see test_api.py) so they don't need a real MLflow server.

---

## 6. Job 2: Train-and-Evaluate

```yaml
train-and-evaluate:
  needs: test                # only runs after Job 1 passes
  if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
```

**Why gate on `main` and non-PR?** Training takes up to 60 minutes. Running it on every PR would:
- Slow down PR review
- Waste compute/money
- Run training without knowing the PR will be merged

Only run expensive operations on the main branch.

```yaml
    - name: Patch params.yaml for CI
      run: |
        python - <<'EOF'
        import yaml
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        params["mlflow"]["tracking_uri"] = "http://localhost:5000"  # not http://mlflow:5000
        with open("params.yaml", "w") as f:
            yaml.dump(params, f)
        EOF
```

**Why patch?** In Docker Compose, the service name `mlflow` resolves to the MLflow container IP. In GitHub Actions, there's no Docker network — `mlflow` hostname doesn't exist. The MLflow server runs on `localhost:5000` in CI. The code reads `tracking_uri` from `params.yaml`, so we must patch it.

```yaml
    - name: Start MLflow server
      run: |
        mlflow server --host 0.0.0.0 --port 5000 \
          --backend-store-uri sqlite:///mlruns/mlflow.db \
          --default-artifact-root ./mlruns/artifacts &
        # Wait for it to be ready (retry loop up to 90 seconds)
        for i in {1..30}; do
          if curl -sf http://localhost:5000/health; then break; fi
          sleep 3
        done
```

**The `&` runs MLflow in the background.** The retry loop is essential — if you immediately run training, MLflow might not be ready yet (race condition).

```yaml
    - name: Evaluate model
      id: evaluate
      run: |
        python src/evaluate.py
        ACCURACY=$(python -c "
        import json
        m = json.load(open('reports/final_metrics.json'))
        print(m.get('accuracy', 0))
        ")
        echo "accuracy=${ACCURACY}" >> $GITHUB_OUTPUT  # set output variable
```

`$GITHUB_OUTPUT` passes values between steps. The next step reads `${{ steps.evaluate.outputs.accuracy }}`.

```yaml
    - name: Check accuracy threshold
      run: |
        python - <<EOF
        threshold = float("${{ github.event.inputs.accuracy_threshold || '0.87' }}")
        accuracy  = float("${{ steps.evaluate.outputs.accuracy }}")
        if accuracy < threshold:
            exit(1)   # non-zero exit = step failure = job failure
        EOF
```

This is a **quality gate** in CI/CD. The pipeline fails if accuracy is below threshold. The Docker image is NOT built if the model is bad.

---

## 7. Job 3: Docker Build

```yaml
docker-build:
  needs: train-and-evaluate

  steps:
    - name: Download ML artifacts from training job
      uses: actions/download-artifact@v4
      with:
        name: ml-artifacts    # downloads reports/ and models/ from Job 2

    - name: Log in to DockerHub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}  # from repo secrets
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Extract Docker metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        tags: |
          type=sha,prefix=sha-        # sha-a1b2c3d4
          type=ref,event=branch       # main
          type=raw,value=latest       # latest

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha          # GitHub Actions cache for layer caching
        cache-to: type=gha,mode=max

    - name: Security scan (Trivy)
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.IMAGE_NAME }}:latest
        severity: HIGH,CRITICAL
        exit-code: "0"   # report but don't fail the job
```

### Docker Tags:
```
omer022/mlops-news-classifier:sha-a1b2c3d    ← immutable (git SHA)
omer022/mlops-news-classifier:main           ← branch tag
omer022/mlops-news-classifier:latest         ← always points to newest main build
```

**Why tag by SHA?** The `latest` tag is mutable — it changes with every push. For production deployment, you'd use the SHA tag to ensure you deploy an exact, immutable version.

**Layer caching (`cache-from: type=gha`):** Docker builds are layer-by-layer. If `requirements.txt` hasn't changed, the pip install layer is reused from the previous build. GitHub Actions cache stores these layers between runs → 5-minute builds instead of 15.

---

## 8. Secrets & Environment Variables

```yaml
env:
  PYTHON_VERSION: "3.10"
  MLFLOW_TRACKING_URI: "http://localhost:5000"
  IMAGE_NAME: omer022/mlops-news-classifier
```

These are workflow-level env vars — available to all jobs.

**GitHub Secrets:** `${{ secrets.DOCKERHUB_USERNAME }}` are repo-level secrets stored encrypted in GitHub. Never in code. Set at: Repo Settings → Secrets and Variables → Actions.

**Workflow inputs (`workflow_dispatch.inputs`):**
```yaml
workflow_dispatch:
  inputs:
    force_train:
      description: "Force model retraining"
      default: "false"
    accuracy_threshold:
      description: "Min accuracy to register model (0-1)"
      default: "0.87"
```

Allows manual triggering with custom parameters. A team lead can manually trigger retraining with `accuracy_threshold: 0.90` for a stricter release.

---

## 9. Production Perspective

**What this repo implements:**
- 4-job sequential pipeline with quality gates
- Synthetic data for fast CI
- Real MLflow training on main branch
- Docker build and push to DockerHub
- Trivy security scan
- GitHub step summary

**What a production system would add:**
- **Branch protection rules**: Require CI to pass before merging to main
- **Multiple environments**: dev/staging/prod — deploy to staging first
- **Canary deployment**: Route 5% of traffic to new model, monitor, then 100%
- **Rollback automation**: If model degrades in production, auto-rollback
- **Secrets rotation**: DOCKERHUB_TOKEN expires and must be rotated
- **Slack/PagerDuty notifications**: Not just GitHub step summary
- **Test result publishing**: JUnit XML reports for PR test summary view
- **Coverage reporting**: Codecov integration

---

## 10. Interview Questions

**Q1 (Beginner): What is the difference between CI and CD?**
- Strong: "CI (Continuous Integration) runs automated tests on every code change. CD (Continuous Delivery/Deployment) takes it further — after CI passes, the application is automatically packaged, and optionally deployed. In this project, CI is Job 1 (tests). CD is Jobs 2 and 3 (train, evaluate, Docker build, push)."

**Q2 (Intermediate): Why does Job 2 (train) only run on the `main` branch and not on PRs?**
- Strong: "Training takes up to 60 minutes and requires downloading data, running MLflow, etc. Running this on every PR would slow down development velocity significantly. PRs only need fast feedback — the unit tests in Job 1 catch most issues in 5 minutes. Training runs after merging to confirm the main branch is in a deployable state."

**Q3 (Intermediate): Why does the CI script patch params.yaml before training?**
- Strong: "params.yaml has `tracking_uri: http://mlflow:5000` — the Docker Compose service hostname. In GitHub Actions, there's no Docker network, so `mlflow` doesn't resolve. The CI runs MLflow directly on `localhost:5000`. Since the Python code reads the URI from params.yaml (not an env var), the CI must patch the file before running training."

**Q4 (Advanced): Describe the quality gate in this CI/CD pipeline.**
- Strong: "There are two quality gates: First, the test suite (Job 1) — all 71 pytest tests must pass for Job 2 to even run. Second, the accuracy threshold check in Job 2 — if the trained model's accuracy is below 0.87 (configurable via workflow_dispatch input), the step exits with code 1, failing the job, and Job 3 (Docker build) never runs. This prevents a degraded model from being built into a container and deployed."

---

## Must Know
- CI: automatic testing on every push; CD: automatic deployment after CI
- GitHub Actions: workflow file defines jobs, jobs have steps
- `needs:` creates job dependencies (sequential execution)
- `if: github.ref == 'refs/heads/main'` gates expensive jobs to main branch only
- Secrets go in GitHub Secrets, not in YAML files
- Quality gates: non-zero exit code = step failure = job failure = downstream jobs don't run
- Docker layer caching speeds up builds (cache-from: type=gha)

## Should Know
- Why synthetic data in CI (speed + reliability)
- Why `--deselect` is used for specific tests in CI
- How artifacts pass data between jobs (`upload-artifact` / `download-artifact`)
- `$GITHUB_OUTPUT` for passing values between steps
- Docker metadata action for tagging by branch + SHA

## Nice to Know
- Self-hosted runners for expensive GPU training jobs
- GitHub Actions OIDC for keyless AWS/GCP authentication
- Reusable workflows (`.github/workflows/reusable.yml`)
- DVC integration with CI for data versioning checks
