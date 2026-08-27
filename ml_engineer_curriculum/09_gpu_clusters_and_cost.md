# TOPIC 09: GPU Clusters & Cost Optimization

---

## WHY

Renting the wrong GPU for a training run costs 2-3× more than necessary. Understanding VRAM requirements, GPU trade-offs, and cost estimation is a core ML engineering skill.

---

## 1. GPU Selection Framework

The three questions to answer before renting:

1. **Can this GPU fit my model?** (VRAM requirement)
2. **How long will training take?** (throughput × dataset size)
3. **What is the total cost?** (hourly rate × hours)

---

## 2. VRAM Requirements Estimation

### Formula:
```
VRAM_needed = model_parameters × bytes_per_param × multiplier

bytes_per_param:
  fp32: 4 bytes
  bf16: 2 bytes
  int8: 1 byte
  int4: 0.5 bytes (NF4 for QLoRA)

multiplier for training:
  Full fine-tuning (Adam): × 16  (weights + grads + 2 optimizer states, all fp32)
  LoRA in bf16 (Adam): model_bf16 + lora_fp32 × 12
  QLoRA (4-bit + LoRA): model_nf4 + lora_fp32 × 12
```

### Worked examples:

**7B model, full fine-tuning, bf16:**
```
Weights: 7B × 2 bytes = 14 GB (stored in bf16)
Gradients: 7B × 4 bytes = 28 GB (upcast to fp32 for stability)
Adam m: 7B × 4 bytes = 28 GB
Adam v: 7B × 4 bytes = 28 GB
Activations: ~10-30 GB (depends on batch size)
─────────────────────────────
Total: ~108-128 GB → 2× A100 80GB minimum
```

**7B model, LoRA (r=16), bf16:**
```
Frozen weights (bf16): 7B × 2 bytes = 14 GB
LoRA params (fp32): ~20M × 4 bytes = 0.08 GB
LoRA grads (fp32): ~0.08 GB
LoRA Adam states: ~0.16 GB
Activations: ~6-12 GB
─────────────────────────────
Total: ~20-26 GB → 1× A100 40GB or H100 80GB
```

**7B model, QLoRA (4-bit + LoRA):**
```
Quantized weights (nf4): 7B × 0.5 bytes = 3.5 GB
LoRA params + optimizer: ~0.3 GB
Activations: ~4-8 GB
─────────────────────────────
Total: ~8-12 GB → RTX 4090 (24GB) or A10 (24GB)
```

### Rule of thumb:
```
Full FT: need VRAM_GPUs × GPU_VRAM >= 16 × model_params_in_bytes
LoRA bf16: need ~3× model_bf16 size
QLoRA:     need ~1.5-2× model_nf4 size
```

---

## 3. GPU Comparison

Note: Prices change frequently by provider and vary by availability. The values below are approximate relative comparisons. Always check current provider prices.

| GPU | VRAM | FP16 TFLOPS | BF16 TFLOPS | NVLink | Best for |
|-----|------|-------------|-------------|--------|----------|
| H100 SXM | 80 GB | 989 | 989 | Yes | Maximum throughput multi-GPU |
| H100 PCIe | 80 GB | 756 | 756 | No | Single-node large model |
| A100 SXM 80GB | 80 GB | 312 | 312 | Yes | Standard multi-GPU workhorse |
| A100 SXM 40GB | 40 GB | 312 | 312 | Yes | 7B LoRA in bf16 |
| L40S | 48 GB | 362 | 362 | No | Inference, some training |
| RTX 4090 | 24 GB | 165 | 165 | No | QLoRA 7B, consumer |
| A10G | 24 GB | 125 | 125 | No | QLoRA 7B, cheap inference |
| RTX 3090 | 24 GB | 71 | N/A | No | Budget QLoRA |

**Key ratios (relative to A100 SXM as baseline):**
- H100 SXM is ~3× faster than A100 for transformer training
- A10G is ~40% of A100 throughput at much lower cost
- RTX 4090 is ~53% of A100 throughput (consumer, no ECC)

**Choosing between them:**
- Multi-GPU SFT/DPO on 7B: A100 80GB SXM with NVLink
- Single-GPU QLoRA on 7B: A10 or 4090 (same VRAM, very different cost)
- Very large model (72B): H100 NVLink for throughput
- Inference serving: L40S or A10 (inference-optimized)

---

## 4. Training Time Estimation

### Tokens per second baseline (approximate):
```
Model: 7B, batch_size=8, seq_len=2048
  A100 80GB (bf16, FlashAttention2): ~2,500-4,000 tokens/sec
  H100 80GB (bf16, FlashAttention2): ~7,000-10,000 tokens/sec
  4090 (bf16): ~1,200-2,000 tokens/sec

LoRA vs full FT throughput:
  LoRA is 10-20% faster (fewer gradient computations)
```

### Time estimation:
```
total_tokens = n_samples × avg_seq_length
training_steps = total_tokens / (effective_batch_size × seq_length)
training_time = training_steps / (steps_per_second)

Example:
  Dataset: 10,000 samples × 1,024 tokens = 10.24M tokens
  Effective batch: 64 (2 × 4 GPUs × 8 accum)
  Tokens per step: 64 × 1024 = 65,536
  Steps per epoch: 10.24M / 65,536 ≈ 156 steps
  Steps per second (4× A100): ~2 steps/sec
  Time per epoch: 156 / 2 ≈ 78 seconds ≈ 1.3 minutes (fast!)
  
  But for 100K samples × 2 epochs:
  Steps: 100K × 2048 × 2 / 65,536 = 6,250 steps
  Time: 6,250 / 2 = 3,125 seconds ≈ 52 minutes
```

---

## 5. Cost Calculation

### Training cost:
```python
def estimate_training_cost(
    n_samples: int,
    avg_seq_len: int,
    n_epochs: int,
    effective_batch: int,
    tokens_per_second: float,    # per GPU
    n_gpus: int,
    gpu_hourly_cost: float,      # check current provider prices
    overhead_factor: float = 1.2  # 20% for setup, eval, reruns
) -> dict:
    total_tokens = n_samples * avg_seq_len * n_epochs
    total_seconds = total_tokens / (tokens_per_second * n_gpus)
    total_hours = total_seconds / 3600
    total_cost = total_hours * gpu_hourly_cost * n_gpus * overhead_factor
    return {
        "total_hours": round(total_hours, 2),
        "gpu_hours": round(total_hours * n_gpus, 2),
        "estimated_cost": round(total_cost, 2),
    }

# Example (use current provider prices when running):
result = estimate_training_cost(
    n_samples=10_000, avg_seq_len=1024, n_epochs=2,
    effective_batch=64,
    tokens_per_second=3000,   # A100 benchmark
    n_gpus=4,
    gpu_hourly_cost=YOUR_PROVIDER_PRICE,  # check current prices
    overhead_factor=1.3
)
print(result)
```

**Always add overhead:**
- Debugging runs: 2-3 short runs before the real one
- Eval runs: evaluation adds ~10-15% to compute
- OOM reruns: you WILL hit OOM at least once
- 20-30% overhead factor is conservative

---

## 6. Checkpoint Storage Cost

```
Checkpoint size per save:
  Full model (7B bf16): 14 GB
  LoRA adapter: ~100-200 MB
  Optimizer state (full FT): 56 GB per checkpoint
  Optimizer state (LoRA, fp32): ~400 MB per checkpoint

Checkpoint strategy:
  Save every 100 steps, keep last 3 → max 3 checkpoints stored

Cloud storage cost:
  S3 / GCS: check current prices per GB-month
  Typically cheaper than GPU storage

Example:
  3 LoRA checkpoints × (200MB adapter + 400MB optimizer) = 1.8 GB
  This is negligible vs GPU cost
  
  Full FT: 3 checkpoints × (14GB + 56GB) = 210 GB
  This is significant — use lifecycle rules to delete old checkpoints
```

---

## 7. Cost Reduction Strategies

```
1. SPOT/PREEMPTIBLE INSTANCES
   - 60-80% cheaper than on-demand
   - Risk: job may be interrupted
   - Mitigation: checkpoint every 50-100 steps, resume on interruption
   - Best for: long training runs with checkpointing

2. CHOOSE THE RIGHT GPU
   - For QLoRA 7B: A10 >> A100 in cost efficiency
   - Don't rent an H100 for a 2B model fine-tune
   
3. USE SMALLER MODELS WHERE POSSIBLE
   - 2B fine-tuned >> 7B zero-shot for narrow tasks
   - SmolVLM-2B can outperform Qwen2-VL-7B on specific task domains

4. GRADIENT ACCUMULATION TO MAXIMIZE GPU UTILIZATION
   - Keep GPU memory > 80% utilized
   - Idle GPU memory = wasted money

5. COMPILE THE MODEL
   torch.compile(model, mode="reduce-overhead")
   - 10-30% faster on A100/H100 (less wallclock time = less cost)

6. FLASH ATTENTION 2
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
   - 30-50% faster for long sequences

7. PACK SHORT SEQUENCES
   - Padding short sequences wastes computation
   - Pack multiple samples into one sequence to fill context window
   - DataCollatorWithFlattening in TRL handles this
```

---

## 8. GPU Utilization Checklist

Run this before committing to a long training run:

```bash
# Start a 50-step training run and monitor
nvidia-smi dmon -s u -d 1 | head -60

# Target:
GPU utilization: > 85%    ← GPU is being used
Memory util: > 80%        ← memory is being used (not under-batching)
Power: > 80% of TDP       ← correlated with actual compute

# If GPU util < 70%:
→ DataLoader is bottleneck: increase num_workers, pin_memory=True
→ CPU preprocessing too slow: preprocess and cache offline
→ Batch size too small: increase per_device_batch_size
```

---

## 9. Interview Questions

**Q1 (Intermediate): How would you estimate the cost of fine-tuning a 7B VLM on 50,000 samples?**
Strong: "First estimate VRAM: LoRA bf16 needs ~20GB → 1× A100 40GB or a cheaper alternative. Then estimate throughput: A100 with batch=2, seq_len=1024, LoRA gets ~3,000 tokens/sec. Total tokens: 50K × 1024 × 2 epochs = 102M tokens. Training time: 102M / 3,000 = 34,000 seconds ≈ 9.4 hours. Add 30% overhead → ~12 hours. Multiply by current GPU price (check provider at time of planning). This is an order-of-magnitude estimate — I'd run a 10-step timing benchmark before committing."

**Q2 (Advanced): How would you reduce training cost by 50%?**
Strong: "Five levers: First, QLoRA instead of LoRA bf16 — 4-bit base model reduces VRAM enough to use a cheaper GPU (A10 instead of A100). Second, spot instances — 60-80% cheaper, with checkpoint-based recovery. Third, compile the model with torch.compile — 10-30% faster. Fourth, FlashAttention 2 — 30-50% faster for long sequences. Fifth, sequence packing (DataCollatorWithFlattening) — eliminates padding waste in batches with variable lengths. Combining spot + QLoRA + Flash Attention can realistically reduce cost by 60-70%."

---

## MUST KNOW Summary
- VRAM rule: Full FT ≈ 16× model_bytes; LoRA bf16 ≈ 3×; QLoRA ≈ 1.5-2×
- Training cost = GPU_hourly × GPU_count × hours × overhead
- Always add 20-30% overhead for debugging and reruns
- Use spot instances + checkpointing for long runs (60-80% savings)
- FlashAttention 2 + torch.compile together give 40-60% speedup
- Monitor GPU utilization — < 70% means data loading bottleneck
- GPU prices change: always check current provider prices, don't memorize numbers
