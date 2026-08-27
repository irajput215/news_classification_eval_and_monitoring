# TOPIC 07: Honest Benchmarking Against Frontier Models

---

## WHY

Claiming "our model matches GPT-4V on X" is easy to inflate. Honest benchmarking is what separates engineering integrity from marketing. Interviewers will ask how you compare models — your answer must include controls.

---

## 1. The Core Principle

**Comparing models fairly requires identical treatment.**

Every variable that could influence quality must be controlled:
- Same prompts
- Same preprocessing
- Same context length
- Same evaluation criteria
- Same sampling parameters
- Same number of samples

If any of these differ, you're not measuring model quality — you're measuring setup differences.

---

## 2. Controlled Comparison Checklist

```
PROMPT CONTROL:
  ✓ Identical prompt text for all models
  ✓ Model-appropriate chat template applied for each model
  ✓ Same system prompt (or no system prompt for all)
  ✓ Test with 3+ prompt variations and report the variance

IMAGE INPUT CONTROL:
  ✓ Same image source (same file, same URL)
  ✓ Same image resolution input (don't give GPT-4V 1080p while your model gets 448×448)
  ✓ Same preprocessing (or use model's native preprocessing)
  ✓ Record image token counts for each model

GENERATION CONTROL:
  ✓ Temperature: use 0.0 for deterministic comparison
  ✓ Same max_new_tokens
  ✓ No top_p / top_k differences
  ✓ Seed where supported
  ✓ If any model doesn't support temp=0: report this explicitly

EVALUATION CONTROL:
  ✓ Same metric function for all models
  ✓ Same evaluator (if human: same annotators, same rubric)
  ✓ Blind evaluation where possible (evaluator doesn't know which model)
  ✓ Same judge model if using LLM-as-judge

CONTEXT CONTROL:
  ✓ Same context length (don't give GPT-4o 128K context while your model sees 8K)
  ✓ If context differs: document it explicitly as a limitation
```

---

## 3. Why Leaderboard Numbers Are Often Misleading

**Issue 1: Test set contamination**
Many popular benchmarks (MMLU, HumanEval) are on the internet. Models trained on large web corpora may have "seen" the test questions during pre-training.

How to detect: Evaluate on a PRIVATE held-out test set you created and haven't shared publicly.

**Issue 2: Prompt sensitivity**
```
Prompt A: "Answer with a single word: ..."    → Model X: 72%
Prompt B: "Provide a brief answer: ..."       → Model X: 65%
Prompt C: "What is the answer? Be concise: ..."→ Model X: 68%
```
Which number do you report? Honest answer: report all three and the variance.

**Issue 3: System prompt advantage**
Claude and GPT-4 are typically accessed with powerful hidden system prompts. Your open model has no system prompt. This is a genuine advantage for frontier models that benchmarks ignore.

**Issue 4: Tool availability**
GPT-4 with code interpreter can solve math problems your model cannot. Never compare against tool-augmented frontier models unless your model has the same tools.

**Issue 5: Sampling selection bias**
"We cherry-picked examples where our model wins" — this is common and rarely disclosed.

**How to avoid cherry-picking:**
- Pre-register your test set before running any experiments
- Report full distribution of scores (mean, std, percentiles)
- Report disaggregated results by category
- Show the FAILURES, not just the wins

---

## 4. Calling Frontier APIs Correctly

```python
import anthropic
import openai
import time
from typing import Optional

class FrontierModelEvaluator:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        if "claude" in model_name:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif "gpt" in model_name:
            self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, image_b64: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> dict:
        t0 = time.perf_counter()
        try:
            if "claude" in self.model_name:
                content = [{"type": "text", "text": prompt}]
                if image_b64:
                    content = [
                        {"type": "image", "source": {"type": "base64",
                         "media_type": "image/jpeg", "data": image_b64}},
                        {"type": "text", "text": prompt}
                    ]
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": content}]
                )
                text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            elif "gpt" in self.model_name:
                content = [{"type": "text", "text": prompt}]
                if image_b64:
                    content = [
                        {"type": "image_url", "image_url":
                         {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": content}]
                )
                text = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            return {
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "model": self.model_name,
                "error": None
            }
        except Exception as e:
            return {"text": None, "error": str(e), "latency_ms": None,
                    "model": self.model_name}
```

---

## 5. The Benchmark Report

```python
def generate_benchmark_report(comparison: dict, config: dict) -> str:
    """Generate a structured benchmark report."""
    lines = [
        "# Benchmark Report",
        f"Date: {datetime.now().isoformat()}",
        f"Test samples: {config['n_samples']}",
        f"Temperature: {config['temperature']}",
        f"Max tokens: {config['max_tokens']}",
        f"Prompt version: {config['prompt_hash']}",
        f"Dataset: {config['dataset_name']} (hash: {config['dataset_hash']})",
        "",
        "## Results",
        "",
        "| Model | Accuracy | BERTScore | Latency (p50) | Cost/1K req | Failure % | Human Pref |",
        "|-------|----------|-----------|---------------|-------------|-----------|------------|",
    ]
    for model_id, stats in comparison.items():
        acc = stats["accuracy"]
        bs = stats["bert_score"]
        lat = stats["latency_ms"]["p50"]
        cost = stats.get("cost_per_1k_requests", "N/A")
        fail = stats.get("failure_rate", 0.0)
        pref = stats.get("human_preference_rate", "N/A")
        lines.append(
            f"| {model_id} | "
            f"{acc['mean']:.3f} [{acc['ci_95'][0]:.3f}, {acc['ci_95'][1]:.3f}] | "
            f"{bs['mean']:.3f} [{bs['ci_95'][0]:.3f}, {bs['ci_95'][1]:.3f}] | "
            f"{lat:.0f}ms | {cost} | {fail:.1%} | {pref} |"
        )
    lines += [
        "",
        "## Statistical Significance",
        "Wilcoxon signed-rank test, p < 0.05 considered significant.",
        "",
    ]
    for comparison_key, p_val in comparison.get("significance", {}).get("accuracy", {}).items():
        sig = "✅ Significant" if p_val["significant"] else "❌ Not significant"
        lines.append(f"- {comparison_key}: p={p_val['p_value']:.4f} → {sig}")

    lines += [
        "",
        "## Known Limitations",
        "- Frontier models may use hidden system prompts not controlled here",
        "- Context length differs: [list per model]",
        "- Image resolution: [list per model]",
        "- Rate limiting: [list delays and retry counts]",
        "",
        "## Cost Analysis",
        "",
    ]
    return "\n".join(lines)
```

---

## 6. Cost and Latency in Benchmark Reports

Always report per-request cost alongside quality metrics:

```
Cost comparison example:

Model              | Quality (Acc) | Latency (p50) | Cost/1K req
-------------------|--------------:|-------------:|-----------:
GPT-4o             | 0.847         | 1,200ms       | $2.50
Gemini 1.5 Flash   | 0.821         | 400ms         | $0.075
Qwen2-VL-7B-Instruct| 0.794        | 180ms         | $0.002  (self-hosted)
Your fine-tuned 7B | 0.831         | 180ms         | $0.002  (self-hosted)

Key insight: your fine-tuned 7B achieves 0.831 vs GPT-4o's 0.847
at 1/1250th the cost. This is the story you want to tell.
```

**Cost formula for self-hosted models:**
```
GPU hourly cost: depends on provider (check at time of comparison)
Requests per hour: throughput × 3600
Cost per request: GPU_hourly_cost / requests_per_hour

Example:
  GPU cost: (your actual provider cost)
  Throughput: 150 req/hour (at your load)
  Cost/req: GPU_cost / 150
```

Note: GPU costs change frequently. Always state the date and provider when reporting.

---

## 7. Avoiding Cherry-Picking — Protocol

```
BEFORE running experiments:
  1. Write down your hypothesis: "Our fine-tuned model will outperform baseline on X"
  2. Define your test set (lock it, hash it)
  3. Define your metrics (write them down before seeing results)
  4. Define your success criteria ("accuracy > 0.80 on the test set")

DURING evaluation:
  5. Run all models on ALL samples (not just the easy ones)
  6. Record failures, not just successes
  7. Use blind evaluation where possible

WHEN REPORTING:
  8. Report the full distribution (mean ± CI), not just the peak
  9. Report disaggregated results by difficulty/category
  10. Include qualitative error analysis (what does the model get wrong?)
  11. Disclose any limitations (context length, image resolution, etc.)
```

---

## 8. Interview Questions

**Q1 (Intermediate): How would you fairly benchmark your 7B fine-tuned VLM against GPT-4V?**
Strong: "Control everything: same prompts (with model-appropriate templates), same images at equivalent resolution, same max output tokens, temperature 0 for determinism. For the test set, use a dataset I created — not something publicly available (contamination risk). Run 200+ samples for statistical power. Report per-sample scores, confidence intervals, and a Wilcoxon test for significance. Also report latency, cost per request, and failure rate — because 'matching GPT-4V quality at 1/1000th the cost' is often the real story."

**Q2 (Advanced): A colleague reports your model achieves 91% on VQAv2. Why might you be skeptical?**
Strong: "Three red flags: First, VQAv2 is a public benchmark and may have leaked into training data — check if the training set was filtered against it. Second, 91% is extremely high — state-of-the-art closed models are around 85-86%. Third, what's the evaluation setup? Some implementations report 'yes/no accuracy' which inflates numbers vs exact match. I'd want to see: the exact evaluation code, the prompt format, the test split used (validation vs test-dev), generation parameters, and ideally results on a private held-out test set."

---

## MUST KNOW Summary
- Control all variables: prompt, preprocessing, max tokens, temperature
- Report CIs, not just point estimates
- Test set contamination is real and common in public benchmarks
- Always report cost + latency alongside quality
- Report failure rates (what percentage of samples failed/timed out)
- Pre-register test set before running experiments to avoid cherry-picking
- "Our model at 1/1000th the cost" is often the real competitive story
