# TOPIC 11: Production Model Serving

---

## WHY

A fine-tuned model sitting in a checkpoint is worth nothing. Serving is where ML becomes a product. Serving is also where most ML systems fail — OOM, latency spikes, throughput collapse, cold starts.

---

## 1. The Architecture

```
Client (browser / mobile / service)
    ↓ HTTP/gRPC request
Load Balancer (Nginx / ALB)
    ↓
API Layer (FastAPI / Flask)
    ↓ request queue
Model Server (vLLM / TGI / SGLang)
    ↓
GPU Worker (inference computation)
    ↓
KV Cache (attention keys + values)
    ↓ response
API → Load Balancer → Client
```

---

## 2. The KV Cache

The most important concept in LLM serving.

### What it is
During autoregressive generation, the model processes all previous tokens at each step. Without caching, this is O(n²) computation.

**Without KV cache:**
```
Step 1: process [prompt]               → compute attention for T tokens
Step 2: process [prompt, token_1]      → recompute attention for T+1 tokens
Step 3: process [prompt, token_1, token_2] → recompute for T+2 tokens
...
```

**With KV cache:**
```
Step 1: process [prompt]                  → compute K,V for T tokens → store in cache
Step 2: process only [token_1]            → compute K,V for 1 new token → append to cache
Step 3: process only [token_2]            → 1 new token → append to cache
...
```

The cache stores the Key and Value matrices for all previous tokens. Each new step only processes one new token.

### Memory cost of KV cache
```
KV cache size per token:
  = 2 × n_layers × n_kv_heads × d_head × bytes_per_element

For 7B Qwen2 (bf16):
  n_layers = 28
  n_kv_heads = 8 (GQA)
  d_head = 128
  bytes = 2 (bf16)
  
  = 2 × 28 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB per token

For 4096 context tokens: 4096 × 112 KB = 448 MB
For 32768 context tokens: 32768 × 112 KB = 3.5 GB
```

This is why long context increases memory significantly. vLLM's PagedAttention manages KV cache in pages (like virtual memory) to avoid fragmentation.

---

## 3. Static vs Continuous Batching

### Static Batching
```
Batch: [request_1, request_2, request_3]
       → all processed together
       → all must finish before next batch starts
       → if request_1 finishes early, GPU waits for request_2 and request_3
```

GPU is idle while waiting for the slowest request. Utilization: ~40-60%.

### Continuous Batching (vLLM, TGI default)
```
Step 1: process [req_1_step_1, req_2_step_1, req_3_step_1]
Step 2: process [req_1_step_2, req_2_step_2, req_3_step_2, NEW_req_4_step_1]
                                             ↑ req_3 finished, new request inserted
Step 3: process [req_1_step_3, req_2_step_3, req_4_step_2, NEW_req_5_step_1]
...
```

As soon as any request finishes, a new one fills its slot. GPU utilization: > 80%.

This is the single most important innovation in LLM serving infrastructure.

---

## 4. vLLM

vLLM (UC Berkeley) is the dominant open-source LLM serving library.

### Key innovations:
1. **PagedAttention** — KV cache managed in pages like virtual memory, eliminates fragmentation
2. **Continuous batching** — described above
3. **CUDA graph capture** — graphs the computation once, replays without Python overhead
4. **Quantization support** — AWQ, GPTQ, FP8

### Running vLLM:
```bash
# Install
pip install vllm

# Serve a model as an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \    # use 2 GPUs
    --gpu-memory-utilization 0.9 \  # use 90% of GPU memory for KV cache
    --port 8000
```

### Calling vLLM:
```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# For text
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{"role": "user", "content": "What is LoRA?"}],
    temperature=0.0,
    max_tokens=512,
)

# For vision (image as base64)
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Describe this image."}
        ]
    }],
    temperature=0.0,
)
print(response.choices[0].message.content)
```

### vLLM for VLMs (multimodal):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=1 \    # max 1 image per request
    --max-model-len 8192
```

---

## 5. TGI (Text Generation Inference — HuggingFace)

```bash
# Docker deployment (easiest for production)
docker run --runtime nvidia --gpus all \
    -v $volume:/data \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-input-length 4096 \
    --max-total-tokens 8192

# Call via HTTP
curl http://localhost:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "Describe this image", "parameters": {"max_new_tokens": 200}}'
```

**TGI strengths:**
- Official HuggingFace support, very well-tested
- Built-in Prometheus metrics
- Flash Attention 2 by default
- Good for single-model deployments

---

## 6. SGLang

SGLang is newer and focuses on structured generation and KV cache reuse.

**Key feature: RadixAttention** — reuses KV cache across requests that share the same prefix (e.g., same system prompt). Reduces compute by 50%+ for multi-turn conversations or same-prompt batches.

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --port 30000 \
    --chat-template qwen2-vl
```

---

## 7. Framework Comparison

| Feature | vLLM | TGI | SGLang |
|---------|------|-----|--------|
| Continuous batching | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ❌ (uses different approach) | ✅ |
| VLM support | ✅ Good | ✅ Good | ✅ Good |
| Structured output | Limited | Limited | ✅ Native |
| Prefix caching | ✅ | Limited | ✅ RadixAttention |
| Tensor parallelism | ✅ | ✅ | ✅ |
| CUDA graphs | ✅ | ✅ | ✅ |
| Ease of use | ★★★★ | ★★★★★ | ★★★ |
| Throughput at scale | ★★★★★ | ★★★★ | ★★★★★ |
| Best for | High-throughput inference | Ease of deployment | Multi-turn, structured gen |

**When to choose:**
- vLLM: Maximum throughput, production scale, most model support
- TGI: Quick production deployment, HuggingFace ecosystem
- SGLang: Structured generation (JSON), multi-turn conversations, KV reuse

---

## 8. Quantization for Serving

Quantization reduces model size and increases throughput.

### AWQ (Activation-aware Weight Quantization) — recommended
```bash
# Quantize model to 4-bit AWQ
pip install autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./Qwen2-VL-7B-AWQ")

# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./Qwen2-VL-7B-AWQ \
    --quantization awq
```

### GPTQ — alternative
Similar to AWQ but older algorithm. Slightly lower quality, good tooling.

### FP8 — fastest on H100
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dtype float8   # requires H100
```

### Quantization impact on quality:
```
Full bf16:  ~92% accuracy (baseline)
AWQ INT4:   ~91% accuracy (-1%)
GPTQ INT4:  ~90% accuracy (-2%)
INT8:       ~91.5% accuracy (-0.5%)
```

### Impact on throughput:
```
Full bf16:  100% (baseline)
AWQ INT4:   ~180% throughput (1.8×)
```

---

## 9. Tensor Parallelism

Split the model across GPUs for large models:

```bash
# 2-GPU tensor parallel
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4  # split across 4 GPUs
```

**How tensor parallelism works:**
Each attention head and FFN layer is split across GPUs:
```
GPU 0: handles attention heads 0-7
GPU 1: handles attention heads 8-15
GPU 2: handles attention heads 16-23
GPU 3: handles attention heads 24-31
```

Each GPU computes its portion, then all-reduce to merge. More GPUs = lower latency for large models, but more communication overhead.

---

## 10. Speculative Decoding

Generate draft tokens fast with a small model, verify with the large model:

```
Small model (Draft): generates 4 tokens quickly (speculative tokens)
Large model (Verifier): accepts or rejects each draft token in ONE pass
                        (much cheaper than generating 4 tokens one by one)
```

Net effect: 2-4× speedup for tasks with predictable text (code, structured output).

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --speculative-model Qwen/Qwen2-1.5B-Instruct \  # draft model
    --num-speculative-tokens 5
```

---

## 11. Cold Starts and Autoscaling

**Cold start:** Time from "request arrives" to "first token returned" when the model isn't loaded.
For a 7B model: ~15-30 seconds (model load + CUDA compilation).

**Mitigation:**
1. Keep at least 1 warm instance always running
2. Use model caching (keep model in shared memory across requests)
3. Implement a "warm-up" request at startup

**Autoscaling:**
Scale replicas based on request queue length or GPU utilization:
```yaml
# Kubernetes HPA based on GPU utilization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: 70  # scale when GPU util > 70%
```

---

## 12. Interview Questions

**Q1 (Intermediate): What is continuous batching and why does it improve GPU utilization?**
Strong: "In static batching, you group requests into fixed batches. If one request in the batch generates 50 tokens and another needs 200, the GPU is idle after the short one finishes while waiting for the long one. Continuous batching fixes this: as soon as any request generates its final token, a new waiting request is immediately inserted into the batch at that slot. The GPU is always processing a full batch. This increases GPU utilization from ~40-60% to > 80% and significantly increases throughput."

**Q2 (Intermediate): How would you serve a 7B VLM cheaply?**
Strong: "Four levels: First, quantize to AWQ INT4 — reduces memory from 14GB to 3.5GB and increases throughput by ~1.8×. Second, deploy on A10 (24GB) instead of A100 — same VRAM for INT4 model, much lower hourly cost. Third, use vLLM with continuous batching and GPU memory utilization 0.9 — maximize GPU usage. Fourth, implement request batching in the API layer with a 50ms wait window — collect multiple single-text requests and process as one batch. Combined, these reduce cost per request by 70-80% vs naively serving bf16 on an A100."

**Q3 (Advanced): Explain PagedAttention and why it improves throughput.**
Strong: "In standard attention, the KV cache for each request is allocated as a contiguous block. If request A needs 2048 tokens, you allocate 2048 × 112KB = 224MB for it. When request A finishes, that memory is freed, but the next request might need a different size — leading to fragmentation. PagedAttention manages KV cache in small fixed-size pages (like OS virtual memory). Each request's KV cache is stored in non-contiguous pages. This eliminates fragmentation, allowing many more concurrent requests in the same VRAM. Theoretically, you could pack 3-4× more concurrent requests into the same GPU memory vs contiguous allocation."

---

## MUST KNOW Summary
- KV cache: stores K,V matrices for all previous tokens; O(1) per new token instead of O(n)
- Continuous batching: immediately fill GPU slots as requests complete → > 80% utilization
- vLLM: PagedAttention + continuous batching = gold standard for production LLM serving
- AWQ INT4: ~1.8× throughput, ~1% quality loss vs bf16
- Tensor parallelism: split model across GPUs for models too large for one GPU
- Cold start: 15-30s for 7B model — keep warm instances running
- Speculative decoding: 2-4× speedup with small draft model
