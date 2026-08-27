# TOPIC 06: Building an Evaluation Harness

---

## WHY

A model is only as good as how you measure it. An evaluation harness is the infrastructure that:
- Runs a model against a fixed benchmark
- Computes consistent, reproducible metrics
- Produces comparable results across model versions
- Is designed to be extended and re-run

Without a proper eval harness, you're flying blind. You can't know if your fine-tuning helped.

---

## 1. The Architecture

```
┌─────────────────────────────────────────────────────┐
│                  EVALUATION HARNESS                  │
│                                                     │
│  Input Dataset (test split, never seen during SFT)  │
│       │                                             │
│       ├──► Model A (your fine-tuned VLM)            │
│       ├──► Model B (baseline SFT model)             │
│       └──► Model C (GPT-4V / Gemini API)            │
│                    │                                │
│                    ↓                                │
│           Evaluation Functions                      │
│           (per-metric, deterministic)               │
│                    │                                │
│                    ↓                                │
│                 Metrics                             │
│           (per-sample + aggregate)                  │
│                    │                                │
│                    ↓                                │
│         Statistical Analysis                        │
│         (confidence intervals, significance)        │
│                    │                                │
│                    ↓                                │
│         Benchmark Report (JSON + HTML)              │
└─────────────────────────────────────────────────────┘
```

---

## 2. Core Principles of a Reliable Harness

1. **Fixed test set** — never add to it; never train on it
2. **Deterministic generation** — use `temperature=0.0, do_sample=False` for reproducibility
3. **Identical inputs** — same image preprocessing, same prompt format, same context for all models
4. **Version-locked** — record model checkpoint hash + library versions
5. **Per-sample storage** — save individual predictions so you can inspect failures
6. **Statistical rigor** — report confidence intervals, not just point estimates

---

## 3. Harness Implementation

```python
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

@dataclass
class EvalResult:
    sample_id: str
    prompt: str
    reference: str
    prediction: str
    metrics: Dict[str, float]
    model_id: str
    latency_ms: float

class EvalHarness:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, List[EvalResult]] = {}

    def evaluate_model(
        self,
        model_id: str,
        model_fn: Callable,           # f(prompt, image) → response string
        dataset: List[Dict],
        metrics: List[Callable],
        max_samples: int = None,
    ) -> List[EvalResult]:
        """Run one model on the full dataset."""
        import time
        samples = dataset[:max_samples] if max_samples else dataset
        results = []

        for sample in samples:
            t0 = time.perf_counter()
            prediction = model_fn(sample["prompt"], sample.get("image"))
            latency_ms = (time.perf_counter() - t0) * 1000

            metric_scores = {}
            for metric_fn in metrics:
                metric_name = metric_fn.__name__
                metric_scores[metric_name] = metric_fn(
                    prediction=prediction,
                    reference=sample["reference"],
                    image=sample.get("image"),
                )

            result = EvalResult(
                sample_id=sample["id"],
                prompt=sample["prompt"],
                reference=sample["reference"],
                prediction=prediction,
                metrics=metric_scores,
                model_id=model_id,
                latency_ms=latency_ms,
            )
            results.append(result)

        self.results[model_id] = results
        # Save per-model results
        with open(self.output_dir / f"{model_id}_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

        return results

    def compute_aggregate(self, model_id: str) -> Dict[str, Any]:
        """Compute aggregate stats with 95% confidence intervals."""
        results = self.results[model_id]
        metrics = list(results[0].metrics.keys())
        agg = {}

        for metric in metrics:
            scores = [r.metrics[metric] for r in results]
            n = len(scores)
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            se = std / np.sqrt(n)
            # 95% confidence interval (t-distribution for small samples)
            ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
            agg[metric] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci_95": [round(ci[0], 4), round(ci[1], 4)],
                "n": n,
            }

        latencies = [r.latency_ms for r in results]
        agg["latency_ms"] = {
            "mean": round(np.mean(latencies), 1),
            "p50": round(np.percentile(latencies, 50), 1),
            "p95": round(np.percentile(latencies, 95), 1),
            "p99": round(np.percentile(latencies, 99), 1),
        }
        return agg

    def compare_models(self) -> Dict:
        """Compare all evaluated models with significance testing."""
        comparison = {}
        model_ids = list(self.results.keys())

        for model_id in model_ids:
            comparison[model_id] = self.compute_aggregate(model_id)

        # Pairwise significance tests
        if len(model_ids) >= 2:
            comparison["significance"] = {}
            metrics = list(self.results[model_ids[0]][0].metrics.keys())
            for metric in metrics:
                comparison["significance"][metric] = {}
                for i, m1 in enumerate(model_ids):
                    for m2 in model_ids[i+1:]:
                        scores1 = [r.metrics[metric] for r in self.results[m1]]
                        scores2 = [r.metrics[metric] for r in self.results[m2]]
                        # Wilcoxon signed-rank test (non-parametric, robust)
                        stat, p_value = stats.wilcoxon(scores1, scores2)
                        comparison["significance"][metric][f"{m1}_vs_{m2}"] = {
                            "p_value": round(p_value, 4),
                            "significant": p_value < 0.05,
                        }

        return comparison

    def generate_report(self) -> str:
        """Generate markdown benchmark report."""
        comparison = self.compare_models()
        report = ["# Benchmark Report\n"]

        # Aggregate table
        model_ids = list(self.results.keys())
        metrics = list(self.results[model_ids[0]][0].metrics.keys())

        header = "| Model | " + " | ".join(metrics) + " | Latency (p50) |"
        divider = "|" + "---|" * (len(metrics) + 2)
        report.append(header)
        report.append(divider)

        for model_id in model_ids:
            agg = comparison[model_id]
            row = f"| {model_id} |"
            for metric in metrics:
                m = agg[metric]
                row += f" {m['mean']:.3f} [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}] |"
            row += f" {agg['latency_ms']['p50']}ms |"
            report.append(row)

        return "\n".join(report)
```

---

## 4. Metrics — The Full Reference

### 4.1 Exact Match
```python
def exact_match(prediction: str, reference: str, **kwargs) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())

# Normalized exact match (handles punctuation)
import string
def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def exact_match_normalized(prediction, reference, **kwargs):
    return float(normalize(prediction) == normalize(reference))
```

**Use when:** Short, unambiguous answers (VQA, classification, single-word answers)

### 4.2 BLEU (Bilingual Evaluation Understudy)
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu_score(prediction: str, reference: str, **kwargs) -> float:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    smoothing = SmoothingFunction().method1  # handle short sentences
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
```

**Formula:**
```
BLEU = BP × exp(Σ wₙ × log pₙ)

where:
  BP = brevity penalty (penalizes short generations)
  pₙ = n-gram precision (fraction of n-grams in prediction that appear in reference)
  wₙ = weight for n-gram order (typically 1/4 each for BLEU-4)
```

**Strengths:** Fast, deterministic, language-agnostic
**Weaknesses:** Doesn't measure recall; penalizes valid paraphrases; poor correlation with human judgment for long text

**Use when:** Machine translation benchmarks (industry standard), caption generation

### 4.3 ROUGE
```python
from rouge_score import rouge_scorer

def rouge_l(prediction: str, reference: str, **kwargs) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure

def rouge_1(prediction: str, reference: str, **kwargs) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return scorer.score(reference, prediction)["rouge1"].fmeasure
```

**ROUGE-L:** Longest Common Subsequence (LCS) recall. Measures how much of the reference appears in the prediction.

**Use when:** Summarization, long-form generation, document-level tasks

### 4.4 BERTScore
```python
from bert_score import score as bert_score_fn

def bert_score(prediction: str, reference: str, **kwargs) -> float:
    P, R, F1 = bert_score_fn(
        [prediction], [reference],
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        device="cuda",
    )
    return float(F1[0])  # F1 is usually the reported score
```

**How it works:**
1. Embed both prediction and reference with a contextual model (DeBERTa)
2. Compute cosine similarity between each prediction token and each reference token
3. Precision: for each prediction token, find max similarity to any reference token
4. Recall: for each reference token, find max similarity to any prediction token
5. F1 = harmonic mean of precision and recall

**Why it's better than BLEU/ROUGE:** Captures semantic similarity, not just surface token overlap. "dog" and "canine" score high similarity.

**Use when:** Open-ended generation, paraphrase-heavy tasks, creative writing

### 4.5 CLIP Similarity (for VLMs)
```python
import torch
import clip

device = "cuda"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def clip_text_image_similarity(image, text: str, **kwargs) -> float:
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze()
    return float(similarity)
```

**Use when:** Evaluating whether a VLM's description is visually grounded; image captioning quality

### 4.6 WER (Word Error Rate) — for ASR
```python
from jiwer import wer

def word_error_rate(prediction: str, reference: str, **kwargs) -> float:
    return wer(reference, prediction)

# WER formula:
# WER = (S + D + I) / N
# S = substitutions, D = deletions, I = insertions, N = reference words
```

### 4.7 Pass@k (for code generation)
```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n: total samples generated
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))
```

**Use when:** Code generation — measures probability that at least 1 of k attempts passes tests

### 4.8 Win Rate (pairwise)
```python
def win_rate(model_a_scores: List[float], model_b_scores: List[float]) -> float:
    """Fraction of samples where model A is preferred over model B."""
    wins = sum(1 for a, b in zip(model_a_scores, model_b_scores) if a > b)
    ties = sum(1 for a, b in zip(model_a_scores, model_b_scores) if a == b)
    return (wins + 0.5 * ties) / len(model_a_scores)
```

### 4.9 Judge Model (LLM-as-Judge)
```python
import anthropic

def judge_model_score(prompt: str, prediction: str, reference: str = None, **kwargs) -> float:
    """Use Claude to rate a response 1-5."""
    client = anthropic.Anthropic()
    judge_prompt = f"""Rate the following response on a scale from 1 to 5.
1 = Poor, 2 = Below average, 3 = Average, 4 = Good, 5 = Excellent

Task: {prompt}
Response to rate: {prediction}
{"Reference answer: " + reference if reference else ""}

Respond with only a single integer (1-5)."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    try:
        score = int(message.content[0].text.strip())
        return (score - 1) / 4.0  # normalize to [0, 1]
    except ValueError:
        return 0.5  # fallback
```

---

## 5. Bootstrap Confidence Intervals

Why CIs matter: if Model A scores 0.72 and Model B scores 0.74 on 100 samples, is that difference real?

```python
import numpy as np

def bootstrap_ci(scores: List[float], n_boot: int = 10000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for mean."""
    boot_means = []
    scores_arr = np.array(scores)
    for _ in range(n_boot):
        sample = np.random.choice(scores_arr, size=len(scores_arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lower, upper)

# Usage:
accuracy_scores = [1.0, 0.0, 1.0, 1.0, 0.0, ...]  # per-sample
ci_lower, ci_upper = bootstrap_ci(accuracy_scores)
print(f"Accuracy: {np.mean(accuracy_scores):.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Rule of thumb for reporting:** If confidence intervals of two models overlap, the difference is NOT statistically significant. Always report CIs, never just point estimates.

---

## 6. Why Benchmarks Can Be Misleading

1. **Test set contamination**: If training data includes the test set (even indirectly via web scraping), accuracy is inflated. This is suspected for several MMLU/HumanEval results.

2. **Benchmark overfitting**: Models trained heavily on MMLU-adjacent data score high on MMLU but don't generalize.

3. **Prompt sensitivity**: A model might score 70% with one prompt format and 50% with another. Always test multiple prompt variations.

4. **Single-metric gaming**: A model optimized for BLEU generates short, n-gram-dense outputs that score high on BLEU but are poor by human judgment.

5. **Sampling settings matter**: Temperature 0.0 vs 0.7 can change accuracy by 5-10% on the same model.

6. **Context length**: GPT-4 with full context may get more information than your model with truncated context.

**How to report responsibly:**
- Report the exact prompt template used
- Report generation parameters (temperature, top_p, max_tokens)
- Report confidence intervals
- Report failure cases (qualitative analysis of errors)
- Report performance disaggregated by category/difficulty

---

## 7. Repositories

```
LM-Evaluation-Harness (EleutherAI):
  https://github.com/EleutherAI/lm-evaluation-harness
  This is the gold standard for LLM benchmarking
  File: lm_eval/tasks/ ← how individual tasks are defined
  Run: lm_eval --model hf --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct --tasks mmlu

VLMEvalKit:
  https://github.com/open-compass/VLMEvalKit
  For vision-language benchmarks
  Run: python run.py --data MMMU_DEV_VAL --model Qwen2-VL-7B

MMMU benchmark:
  https://mmmu-benchmark.github.io/
  Multi-discipline multimodal understanding — the standard VLM benchmark
```

---

## 8. Interview Questions

**Q1 (Intermediate): Why is BERTScore better than BLEU for evaluating VLM responses?**
Strong: "BLEU measures n-gram overlap — if the prediction uses synonyms or paraphrases, it gets a low BLEU score even if the meaning is correct. BERTScore embeds both prediction and reference with a contextual model and computes cosine similarity in embedding space. 'A dog is running' and 'A canine is sprinting' get high BERTScore but low BLEU. For VLM tasks like captioning and open-ended VQA, responses vary in phrasing but share meaning — BERTScore captures this better."

**Q2 (Advanced): How do you ensure your evaluation is reproducible across weeks/models?**
Strong: "Five mechanisms: First, fix the test set — hash it and store the hash. Alert if it changes. Second, fix generation parameters — temperature=0, do_sample=False, seed=42, record max_new_tokens. Third, fix preprocessing — same image resizing, same normalization, same chat template. Fourth, fix metric library versions — pin versions in requirements.txt and hash the metric code. Fifth, store per-sample predictions — so you can debug retroactively and add new metrics without re-running the model. I store results as JSON with model_id, checkpoint_hash, library_versions, and timestamp."

---

## MUST KNOW Summary
- Evaluation harness: fixed test set, deterministic generation, per-sample storage
- BLEU: n-gram precision (fast but misses paraphrases)
- ROUGE-L: LCS-based recall (good for summarization)
- BERTScore: semantic similarity in embedding space (best for open-ended)
- CLIP similarity: visual-text grounding (VLM-specific)
- WER: (S+D+I)/N, lower is better (ASR tasks)
- Win rate: pairwise preference fraction
- Always report confidence intervals: single point estimates are misleading
