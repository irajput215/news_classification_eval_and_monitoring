# TOPIC 12: Testing & Error Handling

---

## 1. What is it?

**Testing in MLOps** means automated verification of three distinct things:
1. **Data tests**: Does the input data meet quality standards?
2. **Model tests**: Does the model produce valid, shaped outputs with reasonable values?
3. **API tests**: Does the serving layer return correct responses to valid and invalid requests?

**Error handling** means gracefully managing failures with fallbacks, clear error messages, and correct HTTP status codes — so failures are diagnosable and recoverable.

---

## 2. Why do we need it?

Without tests, every code change is a gamble. ML code is especially tricky:
- A bug in preprocessing might not cause an exception — it silently produces wrong features
- A wrong model output type might not crash — it returns a number that looks valid but is semantically wrong
- An API that returns 200 for invalid inputs will silently corrupt downstream systems

**71 tests** in this project provide a safety net. When you change any code, tests tell you within minutes if you've broken something.

---

## 3. Where is it used in THIS repository?

| File | What it tests | # of tests |
|------|--------------|------------|
| `tests/test_data.py` | Schema, nulls, split integrity, type checking | ~20 |
| `tests/test_model.py` | Model loading, prediction shapes, probability validity, performance | ~20 |
| `tests/test_api.py` | All API endpoints, status codes, response shapes, edge cases | ~30 |

**Configuration:** `pytest.ini` controls test discovery settings.

---

## 4. test_data.py — Data Quality Tests

```python
# tests/test_data.py

class TestSchema:
    def test_train_required_columns(self, train_df):
        missing = REQUIRED_COLUMNS - set(train_df.columns)
        assert not missing, f"Missing columns in train: {missing}"

class TestDataQuality:
    def test_no_null_text_train(self, train_df):
        null_count = train_df["text"].isnull().sum()
        assert null_count == 0, f"Found {null_count} null values"

    def test_label_label_name_consistency(self, train_df):
        """Each label ID should map to exactly one label_name."""
        mapping = train_df.groupby("label")["label_name"].nunique()
        inconsistent = mapping[mapping > 1]
        assert len(inconsistent) == 0

class TestSplitIntegrity:
    def test_no_overlap_train_test(self, train_df, test_df):
        """No exact text duplicates between train and test."""
        train_texts = set(train_df["text"].values)
        test_texts  = set(test_df["text"].values)
        overlap = train_texts & test_texts
        assert len(overlap) == 0

    def test_all_classes_in_all_splits(self, train_df, val_df, test_df):
        """All classes in train must also appear in val and test."""
        train_classes = set(train_df["label"].unique())
        assert train_classes == set(val_df["label"].unique())
        assert train_classes == set(test_df["label"].unique())
```

**The `test_no_overlap_train_test` test is DESELECTED in CI** because synthetic CI data has only 4 unique text templates repeated 300 times — all splits contain the same texts. This is acceptable for CI speed but would be a real problem in production. The test runs in full training.

---

## 5. test_model.py — Model Quality Tests

```python
# tests/test_model.py

@pytest.fixture(scope="module")
def pipeline():
    pkls = sorted(MODELS_DIR.glob("*.pkl"))
    if not pkls:
        pytest.skip("No model .pkl found — run src/train.py first")
    return joblib.load(pkls[-1])   # load the most recently created model

class TestModelLoading:
    def test_model_is_sklearn_pipeline(self, pipeline):
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_two_steps(self, pipeline):
        assert len(pipeline.steps) == 2  # (tfidf, clf)

    def test_model_has_classes(self, pipeline):
        assert hasattr(pipeline, "classes_")   # needed for label lookup in serve.py
        assert len(pipeline.classes_) >= 2

class TestPredictionShape:
    def test_predict_proba_shape(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        n_classes = len(pipeline.classes_)
        assert proba.shape == (len(SAMPLE_TEXTS), n_classes)

    def test_predict_proba_sums_to_one(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)  # allow tiny floating point error

class TestModelPerformance:
    def test_test_accuracy_above_minimum(self, pipeline, test_df):
        preds = pipeline.predict(test_df["text"])
        acc = accuracy_score(test_df["label"], preds)
        MINIMUM_ACCURACY = 0.60   # sanity check — better than random (0.25)
        assert acc >= MINIMUM_ACCURACY
```

**Why `scope="module"` on fixtures?** Loading the pkl file once for ALL tests in the module is much faster than loading per test. `scope="module"` means the fixture is created once, shared across all tests in the file.

**Why `pytest.skip()` instead of `assert`?** If the model doesn't exist yet (fresh checkout, before training), skipping is correct behavior. Failing with an assertion error would be misleading.

---

## 6. test_api.py — API Endpoint Tests

```python
# tests/test_api.py

def build_mock_pipeline():
    """Build a tiny trained pipeline WITHOUT MLflow or real data."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=200)),
        ("clf",   LogisticRegression(max_iter=500)),
    ])
    pipe.fit(train_texts, train_labels)
    return pipe, label_names

@pytest.fixture(scope="module")
def client(mock_pipeline):
    pipeline, label_names = mock_pipeline
    from src import serve
    
    # Inject mock model directly — bypasses MLflow entirely
    serve.MODEL_STATE.update({
        "pipeline"     : pipeline,
        "model_version": "test-v1",
        "label_names"  : label_names,
        "metrics"      : {"val_accuracy": 0.92},
        "loaded_at"    : "2024-01-01T00:00:00Z",
    })
    
    with patch("src.serve._load_model"):   # prevent real MLflow connection
        with TestClient(serve.app) as c:
            yield c
```

**Key pattern:** The API tests inject a mock model into `MODEL_STATE` and mock the `_load_model()` function. This means:
- Tests run without MLflow running
- Tests run without trained models on disk
- Tests are fast (no model loading) and deterministic

```python
class TestPredictEndpoint:
    def test_predict_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422   # Pydantic validation error

    def test_predict_whitespace_text_returns_422(self, client):
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422   # validator strips and rejects

    @pytest.mark.parametrize("text", [
        "Breaking news: Central bank raises interest rates...",
        "Scientists announce major breakthrough...",
        "National team qualifies for World Cup...",
        "Tech giant reports record quarterly revenue...",
    ])
    def test_predict_various_news_texts(self, client, text):
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        assert 0.0 <= response.json()["confidence"] <= 1.0
```

**`@pytest.mark.parametrize`**: Run the same test with multiple inputs. This tests 4 news texts in one test definition. Much cleaner than 4 separate test functions.

---

## 7. Error Handling in serve.py

```python
# src/serve.py — multiple layers of error handling

# Layer 1: Input validation (Pydantic)
@field_validator("text")
def text_not_empty(cls, v):
    if not v or not v.strip():
        raise ValueError("text field must not be empty")
    # Pydantic converts ValueError to HTTP 422 automatically

# Layer 2: Model not loaded
@app.post("/predict")
async def predict(request: PredictRequest):
    pipeline = MODEL_STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

# Layer 3: Prediction failure
    try:
        proba = pipeline.predict_proba([request.text])[0]
        ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

**HTTP Status Code semantics:**
| Code | Meaning | When to use |
|------|---------|-------------|
| 200 | OK | Successful prediction |
| 400 | Bad Request | Client error (batch > 100) |
| 422 | Unprocessable Entity | Validation error (empty text) |
| 500 | Internal Server Error | Model crashed unexpectedly |
| 503 | Service Unavailable | Model not loaded |

**Error handling in evaluate.py and serve.py (fallback pattern):**
```python
try:
    # Preferred path: load from MLflow Registry
    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
except Exception as e:
    log.warning(f"MLflow load failed ({e}), falling back to local pkl")
    # Fallback: load most recent local pkl
    pkls = sorted((ROOT / "models").glob("*.pkl"))
    pipeline = joblib.load(pkls[-1])
```

This is the **graceful degradation** pattern. The system continues working (with a lesser model source) rather than crashing.

---

## 8. Test Organization Patterns

### Fixture scoping strategy:
```
scope="module"  → fixture created once per test FILE
scope="session" → fixture created once for the ENTIRE pytest run
scope="function" → default: recreated for each test (expensive for models)
```

### Test class organization (used in this repo):
```
class TestSchema:          → schema-related tests
class TestDataQuality:     → quality checks
class TestSplitIntegrity:  → split-related tests
```

### Why class-based tests?
- Logical grouping → easier to run a subset: `pytest tests/test_data.py::TestSchema`
- Shared setup via fixtures
- Clear naming in CI output: "TestSchema::test_train_required_columns PASSED"

---

## 9. Production Perspective

**What this repo implements:**
- 71 tests across 3 categories (data, model, API)
- Mock-based testing (API tests don't need MLflow)
- Scope-optimized fixtures (model loaded once per module)
- Proper HTTP status codes for each error type
- Graceful fallback (MLflow → local pkl)

**What a production system would add:**
- **Integration tests**: Tests that use a real (dev) MLflow instance
- **Load tests**: What happens under 1000 concurrent requests? (Locust, k6)
- **Contract tests**: Verify that the API response shape matches what consumers expect
- **Mutation testing**: Are tests actually catching bugs? (mutmut library)
- **Coverage requirements**: Fail CI if test coverage < 80%
- **Property-based testing**: Random inputs to find edge cases (Hypothesis library)

---

## 10. Interview Questions

**Q1 (Beginner): What are the three categories of tests in this project and why are they separate?**
- Strong: "Data tests validate input quality — schema, nulls, class distribution. Model tests validate model outputs — correct shape, valid probabilities, above-random performance. API tests validate the serving layer — correct status codes, response format, edge case handling. They're separate because they test different failure modes. A passing data test doesn't tell you the model is correct. A passing model test doesn't tell you the API handles empty strings correctly."

**Q2 (Intermediate): Why does test_api.py mock the MLflow connection instead of using a real MLflow server?**
- Strong: "Three reasons: Speed — tests must run in seconds. Isolation — tests shouldn't depend on external services (if MLflow is down, tests would fail for the wrong reason). Determinism — using a real server means test results depend on what's in the registry, which can change between runs. The mock injects a known, controlled model into MODEL_STATE, ensuring tests always use the same model."

**Q3 (Advanced): The `test_no_overlap_train_test` test is deselected in CI but runs in full training. Explain the trade-off.**
- Strong: "The CI uses synthetic data — 4 text templates repeated 300 times — so train and test splits contain identical texts (by design). Running this test in CI would always fail, which is a false failure. The real production training uses AG News with 120,000 unique articles, where overlap would be a genuine data leakage bug. Deselecting in CI is pragmatic — it accepts a known limitation of synthetic data while still testing the real scenario in full training runs."

---

## Must Know
- Three test categories: data tests, model tests, API tests
- Mock external dependencies (MLflow) in fast unit tests
- `pytest.skip()` for missing prerequisites (no model file yet)
- `@pytest.mark.parametrize` for testing multiple inputs with one test
- Graceful degradation: MLflow → local fallback
- HTTP 422 for validation errors, 503 for service unavailable

## Should Know
- `scope="module"` fixtures for expensive setup (model loading)
- `patch()` for mocking Python imports and functions
- Test class organization for grouping related tests
- `--deselect` to skip specific tests in CI without removing them from the test suite

## Nice to Know
- Property-based testing with Hypothesis
- Mutation testing with mutmut
- Load testing with Locust
- Contract testing with Pact
