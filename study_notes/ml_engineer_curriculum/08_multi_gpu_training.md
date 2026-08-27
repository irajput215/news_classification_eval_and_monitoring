# TOPIC 08: Multi-GPU Training

---

## WHY

A 7B VLM fine-tune on a single GPU takes 3+ days. On 4 GPUs it takes ~18 hours. On 8 GPUs, ~9 hours. Multi-GPU training is not optional for serious ML engineering.

---

## 1. From Single GPU to Distributed

```
SINGLE GPU:
  One process, one GPU
  All model parameters on one device
  Batch: 4 samples → gradients → update

DATA PARALLELISM (simplest):
  N processes, N GPUs
  SAME model on every GPU
  Different data on each GPU
  Average gradients across GPUs → synchronized update

FSDP / DeepSpeed ZeRO:
  N processes, N GPUs
  Model is SHARDED across GPUs (each GPU holds 1/N of parameters)
  Enables models too large for any single GPU
```

---

## 2. Core Concepts

### World Size
Total number of processes in the distributed training job.
```python
world_size = int(os.environ["WORLD_SIZE"])  # e.g., 8 for 8 GPUs
```

### Rank
Global process ID (0 to world_size-1). Process 0 is the "master" (handles logging, saves).
```python
rank = int(os.environ["RANK"])  # e.g., 0, 1, 2, ..., 7
```

### Local Rank
GPU index on THIS machine. Important on multi-node setups.
```python
local_rank = int(os.environ["LOCAL_RANK"])  # e.g., 0, 1, 2, 3 (per node)
torch.cuda.set_device(local_rank)
```

### Process Group
All processes must be able to communicate. NCCL (NVIDIA Collective Communications Library) is the backend.
```python
torch.distributed.init_process_group(backend="nccl")
```

---

## 3. DDP (Distributed Data Parallel)

The simplest form of multi-GPU training. Each GPU holds the FULL model.

```
GPU 0: full model, batch [0:8]    → gradients_0
GPU 1: full model, batch [8:16]   → gradients_1
GPU 2: full model, batch [16:24]  → gradients_2
GPU 3: full model, batch [24:32]  → gradients_3
                    ↓
         All-Reduce (average gradients across all GPUs)
                    ↓
         All GPUs update with the SAME averaged gradient
```

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Launch with: torchrun --nproc_per_node=4 train.py

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop looks identical to single GPU
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, batch.labels)
        loss.backward()    # gradients computed locally
        optimizer.step()   # DDP all-reduces grads automatically
        optimizer.zero_grad()
```

**DDP All-Reduce:** After each backward pass, DDP automatically averages gradients across all GPUs via NCCL ring-allreduce. This synchronizes the models so all GPUs have identical weights.

**Limitation:** Each GPU must hold the FULL model + FULL gradients + optimizer states. For 7B models this is ~112GB → DDP alone doesn't help for very large models.

---

## 4. Gradient Accumulation

Simulates a larger batch size without increasing memory:

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / accumulation_steps    # scale loss
    loss.backward()                             # accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                        # update after N batches
        optimizer.zero_grad()
```

**Effective batch size formula:**
```
effective_batch_size = per_device_batch × n_gpus × gradient_accumulation_steps

Example:
  per_device_batch = 2
  n_gpus = 4
  grad_accum = 8
  effective_batch = 2 × 4 × 8 = 64
```

**Important:** Gradient accumulation is NOT the same as increasing batch size on a single GPU:
- With accumulation: sequential, no memory increase
- With true batch size: parallel, memory increases proportionally

---

## 5. Effective Batch Size and Learning Rate

When you increase effective batch size, you should ALSO adjust learning rate:
```
Linear scaling rule: lr_new = lr_base × (effective_batch / base_batch)

Example:
  base: batch=8, lr=1e-4
  new: effective_batch=64 → lr=8e-4

Or: Square root scaling (more conservative):
  lr_new = lr_base × sqrt(effective_batch / base_batch)
  lr_new = 1e-4 × sqrt(8) ≈ 2.8e-4
```

---

## 6. ZeRO (Zero Redundancy Optimizer)

ZeRO is DeepSpeed's memory optimization strategy. It eliminates the redundancy in DDP (each GPU stores full optimizer states).

### ZeRO Stages:

```
STANDARD DDP (no ZeRO):
  Each GPU: Model weights + Gradients + Optimizer states (m, v)
  7B model → each GPU needs ~112 GB

ZERO STAGE 1: Optimizer State Sharding
  Model weights: full copy on each GPU (14 GB)
  Gradients: full on each GPU (14 GB)
  Optimizer states: SHARDED across GPUs (28 GB / n_gpus each)
  4 GPUs: saves 21 GB per GPU (75 GB total instead of 112 GB per GPU)

ZERO STAGE 2: + Gradient Sharding
  Model weights: full copy on each GPU (14 GB)
  Gradients: SHARDED across GPUs (14 GB / n_gpus each)
  Optimizer states: SHARDED (28 GB / n_gpus each)
  4 GPUs: each GPU needs ~28 GB (vs 112 GB)

ZERO STAGE 3: + Parameter Sharding
  Model weights: SHARDED across GPUs (14 GB / n_gpus each)
  Gradients: SHARDED
  Optimizer states: SHARDED
  4 GPUs: each GPU needs ~7 GB (!!) — but gather latency increases
```

**ZeRO-3 trade-off:** Much lower memory but requires communicating weights during forward/backward passes → slower (15-30% throughput reduction).

---

## 7. FSDP (Fully Sharded Data Parallel)

PyTorch's native alternative to DeepSpeed ZeRO-3.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Wrap the model — each transformer layer becomes one FSDP unit
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer, Qwen2VLDecoderLayer}
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
)
```

**FSDP vs ZeRO-3:**
| | FSDP | ZeRO-3 |
|--|------|--------|
| Backend | PyTorch native | DeepSpeed (Microsoft) |
| Integration | Works with HF Accelerate natively | Requires DeepSpeed config file |
| Flexibility | Less configurable | More configurable |
| Stability | More stable | More options but more bugs |
| Recommendation | Use for most cases | Use for very large models or when you need ZeRO-Infinity |

---

## 8. Memory Optimization Techniques

### Gradient Checkpointing (Activation Checkpointing)
Trade compute for memory: don't store all intermediate activations, recompute on backward pass.

```python
model.gradient_checkpointing_enable()
```

**Effect:** Reduces activation memory by ~60-70% at the cost of ~20-30% more compute.

**Caution:** gradient_checkpointing ≠ gradient_accumulation. They solve different problems:
- gradient_checkpointing: reduces activation memory (intermediate tensors during forward pass)
- gradient_accumulation: accumulates gradients across steps to simulate larger batches

### Mixed Precision Training
```python
# BF16 (recommended for H100, A100, better numerical stability than FP16)
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs)
```

Memory savings: fp32 (4 bytes/param) → bf16 (2 bytes/param) = 2× memory reduction for weights and activations.

**BF16 vs FP16:**
| | BF16 | FP16 |
|--|------|------|
| Exponent bits | 8 | 5 |
| Dynamic range | Same as FP32 | Narrower → overflow risk |
| Hardware support | A100, H100 native | All CUDA GPUs |
| Recommendation | Use BF16 on modern GPUs | Use FP16 only if BF16 unavailable |

---

## 9. The HuggingFace Accelerate Way

For most LoRA + VLM fine-tuning, you don't write DDP/FSDP code manually. Use Accelerate:

```python
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP  # or DEEPSPEED or MULTI_GPU (DDP)
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Qwen2VLDecoderLayer
num_machines: 1
num_processes: 4
```

```bash
# Launch with Accelerate (handles distributed setup)
accelerate launch --config_file accelerate_config.yaml train.py
# OR
torchrun --nproc_per_node=4 train.py
```

TRL's SFTTrainer/DPOTrainer/GRPOTrainer work natively with Accelerate.

---

## 10. NCCL and All-Reduce

NCCL is NVIDIA's library for GPU-to-GPU communication.

**All-Reduce:** Compute the sum (or average) of a tensor across all GPUs, and give every GPU the result.

**Ring-AllReduce algorithm:**
```
GPUs: 0, 1, 2, 3 arranged in a ring
Each has gradient tensor G with N elements

Step 1: Each GPU sends N/4 of its gradient to the next GPU
Step 2: Each GPU adds received chunk to its own chunk, sends to next
...after 3 steps: each chunk has been summed by all 4 GPUs
Step N: Broadcast results around the ring

Total data sent: 2 × N × (num_gpus-1) / num_gpus ≈ 2N
This is optimal — you cannot synchronize with less communication
```

**Why NCCL bandwidth matters for training speed:**
- 4× A100s with NVLink: ~600 GB/s bidirectional → gradient sync is fast
- 4× A100s without NVLink (PCIe): ~64 GB/s → gradient sync is a bottleneck
- Always prefer NVLink-connected GPUs for training

---

## 11. GPU Utilization Monitoring

```bash
# Real-time GPU monitoring
watch -n1 nvidia-smi

# More detailed (power, memory, utilization per process)
nvidia-smi dmon -s u

# Using PyTorch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"GPU utilization:      {torch.cuda.utilization()}%")
```

**Target GPU utilization:** > 85% is good. < 70% means you're CPU-bottlenecked (data loading, preprocessing).

**How to improve low utilization:**
- Increase DataLoader `num_workers`
- Use persistent workers: `persistent_workers=True`
- Pre-process images offline
- Increase batch size (if memory allows)
- Use `pin_memory=True` for faster CPU→GPU transfer

---

## 12. Interview Questions

**Q1 (Beginner): What is the difference between DDP and FSDP?**
Strong: "DDP replicates the full model on every GPU. Each GPU processes different data, computes gradients independently, then averages them across GPUs via all-reduce. Every GPU needs enough VRAM for the full model. FSDP shards the model itself — each GPU holds only 1/N of the model parameters, gradients, and optimizer states. Before each computation, FSDP gathers the full shard it needs, computes, then discards the gathered tensors to save memory. This allows training models too large for any single GPU."

**Q2 (Intermediate): Explain the effective batch size formula and why it matters.**
Strong: "effective_batch = per_device_batch × n_gpus × gradient_accumulation. If you have per_device=2, 4 GPUs, accum=8, effective batch is 64. This matters because: (1) the batch size affects gradient noise — larger batches have lower variance, (2) you should scale learning rate proportionally (linear scaling rule) when effective batch size changes, (3) gradient accumulation trades time for memory — 8 steps of batch=2 uses the same memory as batch=2 but gives the gradient quality of batch=16."

**Q3 (Advanced): Why is BF16 preferred over FP16 for modern GPU training?**
Strong: "Both are 2-byte formats and give the same memory savings. The difference is in exponent bits: FP16 has 5 exponent bits (smaller dynamic range), which can cause overflow/underflow during gradient computation — specifically, large gradients 'blow up' and small gradients underflow to 0. This requires loss scaling tricks to compensate. BF16 has 8 exponent bits — same dynamic range as FP32 — so gradients don't overflow or underflow. A100 and H100 have native BF16 tensor cores, so there's no speed penalty. FP16 is only preferred on older GPUs (V100) that don't have native BF16 support."

---

## MUST KNOW Summary
- effective_batch = per_device × n_gpus × grad_accum
- DDP: full model per GPU, gradient all-reduce
- FSDP/ZeRO: model sharded across GPUs → enables huge models
- ZeRO Stage 1: shard optimizer; Stage 2: + gradients; Stage 3: + weights
- gradient_checkpointing: saves activation memory (recompute on backward)
- BF16: preferred over FP16 (same range as FP32, fewer overflow issues)
- NCCL: GPU-to-GPU communication backend; NVLink >> PCIe for bandwidth
- Target GPU utilization: > 85%; if lower, check data loading bottleneck
