# TOPIC 08: Docker & Containerization

---

## 1. What is it?

**Docker** is a containerization technology that packages an application and ALL its dependencies (Python version, libraries, OS tools) into a self-contained unit called a container. A container runs identically on any machine that has Docker installed.

**Key concepts:**
- **Image**: A read-only template (snapshot of filesystem + instructions)
- **Container**: A running instance of an image
- **Dockerfile**: Instructions for building an image
- **Docker Compose**: A tool for running multiple containers as a system

---

## 2. Why do we need it?

**The "it works on my machine" problem:**

Without Docker:
```
Developer: "The model works on my Mac (Python 3.11, scikit-learn 1.2)"
Server: "Python 3.9, scikit-learn 0.24 → AttributeError: Pipeline has no attribute predict_proba"
```

With Docker:
```
Developer builds image → ships image to server → server runs image
IDENTICAL environment. IDENTICAL behavior.
```

**In MLOps specifically:**
- Training environment ≠ serving environment → training-serving skew at the OS level
- Different library versions → different model behavior
- Docker ensures reproducibility at the infrastructure level, not just the code level

---

## 3. Where is it used in THIS repository?

**Files:**
- `Dockerfile` — multi-stage build for the API server
- `docker-compose.yml` — orchestrates 3 services: mlflow, api, (train/monitor as profiles)

---

## 4. The Dockerfile — Multi-Stage Build

```dockerfile
# ─────────────────────────────────────────────
# Stage 1: Builder — install all Python deps
# ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for scipy/scikit-learn compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --user \
        scikit-learn numpy pandas fastapi uvicorn[standard] \
        pydantic mlflow joblib pyyaml matplotlib seaborn evidently

# ─────────────────────────────────────────────
# Stage 2: Runtime — lean production image
# ─────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy ONLY installed packages (not build tools, not gcc)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY params.yaml      .
COPY src/             ./src/
COPY models/          ./models/

RUN mkdir -p reports mlruns

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.serve:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
```

### Why Multi-Stage?

| Without multi-stage | With multi-stage |
|--------------------|-----------------|
| Final image includes gcc, build tools, build artifacts | Final image has ONLY runtime essentials |
| Image size: ~2GB | Image size: ~500MB |
| gcc is a security surface (not needed in production) | Minimal attack surface |
| Slower to pull in production | Faster deployment |

**Stage 1 (builder):** Installs build tools + compiles Python packages (scikit-learn has C extensions that need gcc).

**Stage 2 (runtime):** Copies only the compiled packages from builder. gcc, build-essential, and intermediate artifacts are discarded. This is the image that gets shipped to production.

---

## 5. Code Walkthrough — Key Dockerfile Lines

```dockerfile
FROM python:3.10-slim AS builder
```
- `python:3.10-slim`: Official Python image with minimal OS (no GUI tools, etc.)
- `AS builder`: Names this stage so Stage 2 can copy from it
- **Why 3.10?**: Pinned for reproducibility — `python:3.10` always resolves to the same image

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*
```
- `--no-install-recommends`: Don't install optional packages → smaller image
- `rm -rf /var/lib/apt/lists/*`: Delete apt package cache → reduces layer size
- This is in Stage 1 ONLY — these tools are not in the final image

```dockerfile
COPY --from=builder /root/.local /root/.local
```
- Copies the compiled Python packages from Stage 1 to Stage 2
- `/root/.local` is where `pip install --user` stores packages
- No source code, no build tools — just the packages

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```
- Every 30 seconds, Docker checks if the container is healthy
- `start-period=60s`: Give the app 60 seconds to start before checking (model loading takes time)
- `retries=3`: 3 consecutive failures → container marked as "unhealthy"
- Docker Compose can then restart an unhealthy container

```dockerfile
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", ...]
```
- `0.0.0.0`: Listen on all network interfaces (required in containers — not just localhost)
- If you use `127.0.0.1`, the container's port is unreachable from outside

---

## 6. Docker Compose — Service Orchestration

```yaml
# docker-compose.yml
networks:
  mlops_net:
    driver: bridge  # Private Docker network — services can reach each other by name

volumes:
  mlflow_data:  # Named volume — persists data even if container is recreated

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"   # host:container
    volumes:
      - mlflow_data:/mlruns  # Named volume for persistence
    networks:
      - mlops_net
    healthcheck:
      test: [check http://localhost:5000/health]
      interval: 15s
      start_period: 30s

  api:
    build: { context: ., dockerfile: Dockerfile, target: runtime }
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000   # uses SERVICE NAME, not IP
    volumes:
      - ./models:/app/models:ro          # read-only bind mount for models
      - ./data/processed:/app/data/processed:ro
    networks:
      - mlops_net
    depends_on:
      mlflow:
        condition: service_healthy  # waits until MLflow health check passes
```

### Key concepts:

**Service DNS resolution:** Within `mlops_net`, containers can reach each other by service name:
- `api` container calls `http://mlflow:5000` → resolves to the `mlflow` container's IP
- No hardcoded IPs needed

**`depends_on: condition: service_healthy`:** API container waits until MLflow passes its health check before starting. Without this, the API might start and fail to connect to MLflow (which hasn't finished starting yet).

**Volume types:**
- `- mlflow_data:/mlruns` → Named volume: managed by Docker, persists data, not visible on host filesystem
- `- ./models:/app/models:ro` → Bind mount: maps a host directory to container. `:ro` = read-only (container can't modify your models directory)
- `- mlflow_data:` → Named volumes must be declared at the top level

**Profiles (train and monitor):**
```yaml
  train:
    profiles:
      - train   # Only starts with: docker compose --profile train up train
```
This prevents the training job from running every time. You explicitly opt-in with `--profile train`.

---

## 7. Docker Compose Services Summary

| Service | Image | Ports | Purpose | Lifecycle |
|---------|-------|-------|---------|-----------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.8.1` | 5000 | Experiment tracking server | Always running |
| `api` | Built from `Dockerfile` (runtime stage) | 8000 | FastAPI prediction server | Always running |
| `train` | Built from `Dockerfile` (builder stage) | None | One-shot training job | Profile-gated |
| `monitor` | Built from `Dockerfile` (builder stage) | None | One-shot drift check | Profile-gated |

---

## 8. Production Perspective

**What this repo implements:**
- Multi-stage Docker build (smaller image)
- Docker healthcheck (container self-reporting)
- Named volumes (data persistence)
- Service dependency ordering
- Environment variable injection
- Profile-based service activation

**What a production system would add:**
- **Multi-worker:** `--workers 4` for parallel request handling
- **Container registry:** Push image to ECR/GCR, tag by git SHA
- **Kubernetes (K8s):** Orchestrate multiple replicas, rolling updates, autoscaling
- **Resource limits:** `mem_limit: 512m`, `cpus: 0.5` — prevent one container starving others
- **Non-root user:** Security best practice — run as `USER appuser`, not root
- **Read-only filesystem:** `read_only: true` in compose — prevents container tampering
- **Secrets injection:** Docker Secrets or Kubernetes Secrets, not env vars for credentials

---

## 9. Interview Questions

**Q1 (Beginner): What is a Docker image and how does it differ from a container?**
- Strong: "An image is a read-only, immutable snapshot — like a class definition in OOP. A container is a running instance of that image — like an object. You build an image once (`docker build`) and run it many times (`docker run`). Multiple containers can run from the same image simultaneously."

**Q2 (Intermediate): Why does this project use a multi-stage Dockerfile?**
- Strong: "Stage 1 (builder) needs gcc and build tools to compile scikit-learn's C extensions. But production doesn't need gcc. Stage 2 (runtime) copies only the compiled packages from Stage 1 — no build tools. This reduces the final image size from ~2GB to ~500MB, reduces the attack surface (no compiler in production), and speeds up image pulls in CI/CD."

**Q3 (Intermediate): How does `http://mlflow:5000` work inside the api container?**
- Strong: "Docker Compose creates a private network (`mlops_net`) and registers each service name as a DNS hostname on that network. When the api container resolves `mlflow`, Docker's internal DNS returns the IP of the mlflow container. This service discovery is automatic — you never need to hardcode IPs. This is also why CI fails: in GitHub Actions, there's no Docker network, so `mlflow` doesn't resolve — the CI script patches params.yaml to use `localhost:5000` instead."

**Q4 (Advanced): What security problems exist in this Dockerfile?**
- Strong: "Three main issues: First, the CMD runs as root by default — if the container is compromised, the attacker has root privileges inside the container, which can be escalated in some configurations. Fix: add `RUN useradd -m appuser && USER appuser`. Second, CORS allows all origins (`allow_origins=["*"]`) — in production, restrict to specific domains. Third, the MLflow tracking URI has no authentication — anyone on the same network can submit runs."

---

## Must Know
- Docker packages application + dependencies into a portable container
- Multi-stage builds: compile in stage 1, ship only runtime artifacts in stage 2
- Docker Compose orchestrates multiple containers as a system
- Service names (not IPs) for inter-container communication in Compose networks
- Health checks expose container status to orchestration systems
- Bind mounts vs named volumes distinction

## Should Know
- `depends_on: condition: service_healthy` for startup ordering
- Volume types and when to use each
- Docker profiles for optional services
- ENV vs ARG in Dockerfiles
- Layer caching (COPY requirements.txt BEFORE COPY src/ to cache pip install)

## Nice to Know
- Distroless images for minimal attack surface
- Docker Buildx for multi-platform builds (ARM + AMD64)
- Kaniko for building Docker images inside Kubernetes without Docker daemon
- OCI image specification
