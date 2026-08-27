# TOPIC 07: Model Deployment & FastAPI Serving

---

## 1. What is it?

**Model Deployment** is making a trained model available to handle real-world requests. There are two main patterns:

1. **Online (Real-time) Inference**: A client sends a request → gets a prediction immediately (< 100ms)
2. **Batch Inference**: Hundreds of thousands of inputs are scored overnight, results written to a file/DB

This project implements **online inference** via a FastAPI REST API.

**FastAPI** is a modern Python web framework that:
- Auto-generates OpenAPI documentation (`/docs`)
- Validates request/response shapes using Pydantic
- Supports async request handling
- Has very low latency (comparable to Flask, faster than Django)

---

## 2. Why do we need it?

Without an API, your model is just a `.pkl` file. Users/systems can't interact with it without writing Python and loading the pickle themselves.

A REST API provides:
- Language-agnostic interface (any language can call HTTP)
- Input validation (reject bad requests before they reach the model)
- Health checks (load balancer knows if your service is alive)
- Latency measurement (built into the response)
- Model metadata endpoint (what version, what metrics)

---

## 3. Where is it used in THIS repository?

**File:** `src/serve.py`

**Endpoints:**
```
GET  /health          → health check (is the service alive? is model loaded?)
GET  /model/info      → model version, label names, metrics, loaded_at
POST /predict         → single text → label + confidence + top-k probabilities
POST /predict/batch   → list of texts → list of predictions (max 100)
POST /model/reload    → hot-reload model from MLflow Registry (no restart needed)
```

**API documentation:** Auto-generated at `http://localhost:8000/docs`

---

## 4. How does it work in THIS project?

### Architecture:

```
Client (browser / curl / Python requests)
    |
    | HTTP POST /predict
    | Body: {"text": "Apple stocks rose 5%"}
    ↓
FastAPI (uvicorn ASGI server)
    |
    | Pydantic validates input (text not empty)
    ↓
predict() endpoint function
    |
    | pipeline.predict_proba([request.text])
    ↓
sklearn Pipeline (TF-IDF → LR/SVM)
    |
    | Returns probability array [0.03, 0.85, 0.07, 0.05]
    ↓
Format response: label="Business", confidence=0.85, latency_ms=12.3
    |
    ↓
JSON Response to client
```

### Model loading (at startup):

```python
# MODEL_STATE is a module-level dict — shared across all requests
MODEL_STATE = {
    "pipeline"     : None,
    "model_version": "unknown",
    "label_names"  : [],
    "metrics"      : {},
    "loaded_at"    : None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()  # runs ONCE when server starts
    yield          # server handles requests
    # cleanup here if needed

app = FastAPI(lifespan=lifespan)
```

**Why `lifespan` instead of `startup_event`?** Modern FastAPI uses the lifespan context manager. It loads the model once at startup, not on every request (which would be catastrophically slow).

### The _load_model() function:

```python
def _load_model():
    try:
        model_uri = f"models:/{model_name}/{stage}"  # "models:/news_classifier/Production"
        pipeline = mlflow.sklearn.load_model(model_uri)
        # ... fetch version, metrics from MLflow
        MODEL_STATE.update({"pipeline": pipeline, "model_version": version, ...})
    except Exception as e:
        # MLflow unavailable → fall back to local pkl
        pkls = sorted((ROOT / "models").glob("*.pkl"))
        pipeline = joblib.load(pkls[-1])  # most recent pkl
        MODEL_STATE.update({"pipeline": pipeline, "model_version": f"local:{stem}"})
```

**Key design: double fallback.** MLflow Registry → Local pkl → RuntimeError. This makes the service resilient during MLflow outages.

---

## 5. Code Walkthrough

### Pydantic Request Schema:

```python
# src/serve.py lines 141-150
class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3          # how many top predictions to return

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text field must not be empty")
        return v.strip()             # normalize whitespace
```

| Part | What it does | Why it exists |
|------|-------------|---------------|
| `BaseModel` | Pydantic model | Automatic JSON parsing + validation |
| `text: str` | Required string field | Rejects requests without text |
| `top_k: Optional[int] = 3` | Optional with default | Caller can ask for top-1 or top-4 |
| `@field_validator` | Custom validation | Rejects empty string and whitespace-only input |
| `return v.strip()` | Normalize | Prevents accidental leading/trailing spaces |

### Predict endpoint:

```python
# src/serve.py lines 193-231
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    pipeline = MODEL_STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    proba     = pipeline.predict_proba([request.text])[0]  # shape: (n_classes,)
    label_id  = int(np.argmax(proba))
    label     = MODEL_STATE["label_names"][label_id]
    confidence = float(proba[label_id])

    top_k = min(request.top_k, len(MODEL_STATE["label_names"]))
    top_indices = np.argsort(proba)[::-1][:top_k]
    top_predictions = [
        {"label": MODEL_STATE["label_names"][i], "probability": round(float(proba[i]), 4)}
        for i in top_indices
    ]

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        label=label, label_id=label_id, confidence=round(confidence, 4),
        top_predictions=top_predictions,
        model_version=MODEL_STATE["model_version"],
        latency_ms=round(latency_ms, 2),
    )
```

| Line | What it does | Production concern |
|------|-------------|-------------------|
| `if pipeline is None: raise 503` | Check model loaded | Service starts before model loads (race condition) |
| `time.perf_counter()` | High-precision timer | Monitors latency for alerting |
| `predict_proba([request.text])[0]` | Single text → probability array | The `[0]` unwraps the batch dimension |
| `np.argmax(proba)` | Index of highest probability | The predicted class |
| `np.argsort(proba)[::-1][:top_k]` | Sort by probability, take top-k | Gives ranked predictions |
| Return `model_version` | Attach model version to response | Critical for debugging: which version made this prediction? |
| Return `latency_ms` | Attach latency to response | Can build a latency dashboard from response logs |

---

## 6. Input → Processing → Output

| Stage | Input | Processing | Output |
|-------|-------|------------|--------|
| HTTP request | JSON: `{"text": "Apple stocks..."}` | FastAPI parses + Pydantic validates | `PredictRequest` object |
| Validation | `text` field | `text_not_empty()` validator | Normalized text string |
| Prediction | text string | `pipeline.predict_proba([text])` | Probability array (4,) |
| Label lookup | argmax of probabilities | `label_names[argmax]` | String label |
| Top-k | probability array | sort descending, take top-k | List of {label, probability} |
| Response | All of above | Pydantic validation | JSON response |

**Example response:**
```json
{
    "label": "Business",
    "label_id": 1,
    "confidence": 0.8742,
    "top_predictions": [
        {"label": "Business", "probability": 0.8742},
        {"label": "Sci/Tech", "probability": 0.0853},
        {"label": "World",    "probability": 0.0312}
    ],
    "model_version": "1",
    "latency_ms": 12.34
}
```

---

## 7. Health Check Endpoint

```python
@app.get("/health")
async def health():
    pipeline = MODEL_STATE["pipeline"]
    return {
        "status": "healthy" if pipeline is not None else "model_not_loaded",
        "model_version": MODEL_STATE["model_version"],
        "loaded_at":     MODEL_STATE["loaded_at"],
    }
```

**Why health checks matter in production:**
- **Load balancers** (Nginx, ALB) call `/health` every 30 seconds. If it returns non-200 → remove from rotation
- **Kubernetes** readiness probes: don't send traffic until `/health` returns 200
- **Docker Compose** `healthcheck` field: restart container if health check fails
- **Monitoring dashboards**: Track uptime and model availability

**Distinguishing "service alive" vs "model loaded":** The service can be alive (uvicorn running) but model not yet loaded. This health check exposes both states. A load balancer should only add the instance to rotation when `status: healthy`.

---

## 8. Batch Prediction Endpoint

```python
@app.post("/predict/batch")
async def predict_batch(texts: List[str]):
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")

    probas  = pipeline.predict_proba(texts)   # vectorized — much faster than 100 single calls
    labels  = pipeline.predict(texts)
    ...
```

**Why batch is faster:** sklearn's TF-IDF vectorizer can process a list of texts in one matrix operation. 100 separate HTTP calls have 100x the network round-trip overhead.

**The 100-item limit:** Prevents memory exhaustion if someone sends 10,000 texts in one request. Set based on memory profiling + acceptable latency.

---

## 9. Hot Reload Endpoint

```python
@app.post("/model/reload")
async def reload_model():
    try:
        _load_model()  # re-connects to MLflow, loads current Production model
        return {"status": "reloaded", "model_version": MODEL_STATE["model_version"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Why this matters:** When you promote a new model version to Production in MLflow, the running API still has the old model in memory. Options:
1. Restart the container (causes downtime)
2. Call `/model/reload` (zero downtime hot-swap)

In production, you'd automate this: after successful model promotion, trigger a webhook that calls `/model/reload` on all serving instances.

---

## 10. Production Perspective

**What this repo implements:**
- FastAPI single-instance serving
- MLflow Registry model loading with local fallback
- Health check, model info, single/batch predict, hot reload
- Latency measurement in every response
- Input validation via Pydantic
- CORS middleware

**What a production system would add:**
- **Multiple workers**: `--workers 4` or use Gunicorn as process manager
- **Authentication**: API key header, OAuth2, or JWT
- **Rate limiting**: Prevent DoS (100 req/sec per user)
- **Request logging**: Every request/response logged with request_id for tracing
- **Distributed tracing**: OpenTelemetry integration
- **Async model loading**: Non-blocking model load (current blocking load causes startup delay)
- **Circuit breaker**: If MLflow is down, don't retry every request — open the circuit
- **GPU acceleration**: For deep learning models, route to GPU-enabled workers

**Latency SLAs:**
- p50: < 20ms (half of requests under this)
- p99: < 100ms (99% of requests under this)
- TF-IDF + LR is extremely fast: expect ~5-15ms per request

---

## 11. Interview Questions

**Q1 (Beginner): Why do we need a REST API for model serving instead of just calling the model directly?**
- Strong: "A .pkl file is only useful to Python code. A REST API allows any language (JavaScript, Java, Go) to call the model. It also adds input validation, error handling, health checks, and versioning. In production, applications can call the API without knowing anything about ML or sklearn."

**Q2 (Intermediate): This project loads the model at startup, not per request. Why?**
- Strong: "Loading a sklearn Pipeline with TF-IDF from disk takes 0.5-2 seconds. If we loaded it per request, each prediction would take 2 seconds just for model loading. Instead, `_load_model()` runs once at startup, stores the pipeline in `MODEL_STATE` (a module-level dict shared across all requests). Each request just reads the already-loaded object, which takes microseconds."

**Q3 (Intermediate): What happens if MLflow is down when the serving container starts?**
- Strong: "The `_load_model()` function has a try/except. If MLflow fails (`except Exception`), it falls back to loading the most recently sorted `.pkl` file from the `models/` directory. The `model_version` is set to `local:{filename}` instead of a registry version number. This fallback is why `joblib.dump()` is called after each training run — not just as a backup, but as a required serving fallback."

**Q4 (Advanced): How would you scale this API to handle 10,000 requests per second?**
- Strong: "4 layers: (1) Horizontal scaling — deploy multiple container replicas behind a load balancer (Nginx, ALB). (2) Vertical scaling — this TF-IDF model is CPU-bound; more cores helps. (3) Caching — if the same text appears repeatedly (news headlines), cache results in Redis with a short TTL. (4) Async batching — queue individual requests and process in micro-batches (Triton Inference Server does this). The model itself is fast (~5ms per prediction), so the bottleneck would be network I/O at scale."

---

## Must Know
- FastAPI auto-validates requests via Pydantic models
- Model must be loaded once at startup, stored in-memory, reused per request
- Health check endpoint is essential for production (load balancers + Kubernetes)
- `/predict` returns label, confidence, top-k predictions, model_version, latency_ms
- `model_version` in response enables debugging ("which model made this prediction?")

## Should Know
- `lifespan` context manager vs deprecated `@app.on_event("startup")`
- `predict_proba` vs `predict` — why you need probabilities for confidence scores
- Batch vs single prediction trade-offs
- Hot reload vs container restart for model updates
- Pydantic `@field_validator` for custom validation logic

## Nice to Know
- Triton Inference Server for high-throughput GPU serving
- Seldon Core / BentoML as dedicated model serving frameworks
- ONNX for cross-framework model portability
- gRPC vs REST for high-throughput serving
