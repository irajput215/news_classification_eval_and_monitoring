# TOPIC 12: Low-Cost Inference

---

## 1. The Cost Formula

```
Cost per request = GPU cost per second / requests per second

GPU cost per second = hourly_rate / 3600

Throughput (requests per second) determines everything.
Doubling throughput halves cost per request.
```

## 2. Worked Numerical Example

```
Baseline setup:
  GPU: 1× A100 (check current provider pricing)
  Model: 7B bf16, vLLM, continuous batching
  Throughput: ~150 requests/hour (each request = 200 input + 200 output tokens)
  
  Cost per request = hourly_rate / 150

After optimizations:
  AWQ INT4: 1.8× throughput → 270 req/hr
  Sequence packing: 1.2× more  → 324 req/hr
  Flash Attention 2: already included in vLLM
  
  New cost per request = hourly_rate / 324
  Savings: (1 - 150/324) = 54% cheaper per request
```

Note: Replace `hourly_rate` with current provider pricing when calculating.

## 3. Throughput Formula

```python
throughput_rps = (batch_size × tokens_per_second) / avg_output_tokens

# Example:
# tokens_per_second = 5,000 (A100, 7B AWQ)
# avg_output_tokens = 200
# concurrent_requests = 25 (batch size vLLM maintains)
# throughput ≈ 5000 / 200 = 25 RPS
```

## 4. Every Lever That Reduces Cost

| Lever | Mechanism | Typical savings |
|-------|-----------|-----------------|
| AWQ INT4 | 4-bit weights → smaller model → more batch fit + faster | 40-50% per request |
| Batch size | More requests per GPU step | Linear with batch fill |
| Smaller model | 2B vs 7B → 3× less compute | 70% if quality holds |
| Prompt caching | Reuse KV cache for shared prefixes | 30-50% for same-prompt batches |
| Speculative decoding | Draft + verify → 2-4× for predictable text | 50-75% latency |
| Knowledge distillation | Train small model to mimic large | Quality-dependent |
| GPU routing | Use cheap GPU for simple requests | 30-60% with routing |
| Output length limits | max_new_tokens matters | Linear with token count |
| Prompt compression | Remove redundant prompt tokens | 20-30% input cost |

## 5. Model Routing (Cascade)

Not every request needs the 7B model. Route easy requests to a 2B model:

```python
async def smart_route(request):
    # Fast heuristic: if request is simple (short input, simple question)
    if len(request.text) < 200 and not contains_image(request):
        response = await small_model.generate(request)  # 2B model — cheap
        if response.confidence > 0.85:
            return response
    # Complex request or low confidence → use large model
    return await large_model.generate(request)         # 7B model
```

This can reduce cost by 30-60% if 50%+ of requests are simple.

## 6. Output Length Control

Output tokens cost as much as generating them. Control output length strictly:

```python
# Bad: unlimited output
response = model.generate(prompt, max_new_tokens=2048)

# Good: task-appropriate limits
if task == "classification":
    max_new_tokens = 10       # "Sports" is 1 token
elif task == "short_description":
    max_new_tokens = 100
elif task == "full_caption":
    max_new_tokens = 256
```

For structured output (JSON), constrained decoding ensures the model stops exactly when the JSON is complete.

## 7. Monitoring Cost Per Request

```python
# Add to your vLLM client wrapper
def tracked_generate(client, request):
    response = client.chat.completions.create(**request)
    
    # Track token usage
    usage = response.usage
    metrics.record({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": ...,
    })
    return response
```

Aggregate hourly: `total_tokens_per_hour / tokens_per_dollar` gives your actual cost.

## 8. Interview Question

**Q: How would you calculate and reduce inference cost per request?**
Strong: "Cost per request = GPU hourly rate / requests per hour. To reduce it: First, increase throughput — AWQ INT4 gives 1.8× more throughput at ~1% quality cost. Second, use continuous batching (vLLM) — fills GPU with multiple requests, eliminating idle time. Third, limit output tokens — an unconstrained 7B generating 1000 tokens costs 5× more than one constrained to 200. Fourth, route simple requests to a small model — if 50% of requests can be handled by a 2B model with the same quality, cost halves for those requests. Fifth, use prefix caching for shared system prompts — if all requests share a 500-token system prompt, KV cache eliminates recomputing it."
