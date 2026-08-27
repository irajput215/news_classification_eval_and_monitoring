# TOPIC 02: LoRA & Parameter-Efficient Fine-Tuning

---

## WHY

Full fine-tuning a 7B model requires ~112GB VRAM (7B params × 16 bytes for weights + optimizer states). A single H100 has 80GB. LoRA reduces VRAM to ~20GB for the same model.

LoRA is not a trick. It is the standard approach for fine-tuning LLMs and VLMs.

---

## 1. Why Full Fine-Tuning Is Expensive

For a 7B parameter model:
```
Model weights (bf16):  7B × 2 bytes = 14 GB
Gradients (bf16):      7B × 2 bytes = 14 GB
Optimizer states (AdamW):
  momentum m (fp32):  7B × 4 bytes = 28 GB
  variance v (fp32):  7B × 4 bytes = 28 GB
Activations:          ~10-50 GB depending on batch/seq
─────────────────────────────────────────────
TOTAL:                ~94-134 GB
```

Single H100 (80GB) cannot hold this. You either need multi-GPU or memory tricks.

---

## 2. The LoRA Insight

LoRA (Low-Rank Adaptation) is based on one empirical observation:

> **During fine-tuning, the weight updates ΔW tend to have low intrinsic rank.**

In other words: you don't need to change all 7B parameters. You need to change a small number of "important directions" in weight space.

---

## 3. Mathematical Foundation

A weight matrix W in a transformer layer has shape `(d_out, d_in)`.

Full fine-tuning updates: `W_new = W_orig + ΔW`

ΔW has `d_out × d_in` parameters. For a large layer this is millions of parameters.

**LoRA decomposes ΔW into two smaller matrices:**

```
ΔW = B × A

where:
  A has shape (r, d_in)    — "down projection"
  B has shape (d_out, r)   — "up projection"
  r is the rank (4, 8, 16, 32, 64...)
```

**Parameter count comparison:**
```
Full: ΔW    = d_out × d_in     = 4096 × 4096 = 16,777,216
LoRA: A + B = r × d_in + d_out × r = 8 × 4096 + 4096 × 8 = 65,536

Reduction factor: 16,777,216 / 65,536 = 256×
```

**The forward pass with LoRA:**
```
h = W_orig × x + ΔW × x
  = W_orig × x + (B × A) × x
  = W_orig × x + B × (A × x)
```

The computation: x → A (small projection) → B (small projection) → add to original output.

**Scaling factor (α/r):**
```
ΔW = (α / r) × B × A
```

`α` is a hyperparameter (typically set equal to r, making α/r = 1).
Setting α > r amplifies the LoRA update. Setting α < r shrinks it.
In practice: set `lora_alpha = 2 × lora_r` for a 2× effective learning rate on LoRA weights.

---

## 4. Rank (r)

**What it controls:** How much expressive power the LoRA adapter has.

| r | # params (for 4096×4096 layer) | Use case |
|---|------|---------|
| 4 | ~32K | Tiny adaptation, very low memory |
| 8 | ~64K | Standard, most tasks |
| 16 | ~128K | Better quality for complex tasks |
| 32 | ~256K | Heavy adaptation |
| 64 | ~512K | Near full fine-tune quality |

**Rule of thumb:**
- Start with r=16 for VLM fine-tuning
- If quality is insufficient, double to r=32
- If memory is tight, halve to r=8
- r=4 for quick iteration experiments only

Higher rank = more parameters = slower training + more memory + (sometimes) better results.

---

## 5. Alpha (α)

```python
effective_lr_scaling = lora_alpha / lora_rank
```

- If `lora_alpha = lora_rank`: effective scaling = 1.0 (LoRA learning rate = base lr)
- If `lora_alpha = 2 * lora_rank`: effective scaling = 2.0 (LoRA learns 2× faster)

**Common setting:** `lora_alpha = 2 × lora_rank` (e.g., r=16, alpha=32).

This is often called the "2α trick" — it makes LoRA weights update at 2× the base learning rate without changing the optimizer learning rate.

---

## 6. LoRA Initialization

LoRA matrix initialization is critical:

```python
# A is initialized with Gaussian noise (small)
A ~ N(0, σ²)    # default: σ = 1/sqrt(rank)

# B is initialized to ZERO
B = 0
```

**Why B=0?** At the start of training, ΔW = B × A = 0 × A = 0. The model starts from the pre-trained weights unchanged. This is essential — if you initialize B randomly, you immediately corrupt the pre-trained model at step 0, and training never recovers.

---

## 7. Target Modules

You choose WHICH weight matrices to apply LoRA to.

### Transformer layer anatomy:
```
Self-Attention:
  q_proj:  (d, d_head × n_heads)   ← query projection
  k_proj:  (d, d_kv × n_kv_heads) ← key projection
  v_proj:  (d, d_kv × n_kv_heads) ← value projection
  o_proj:  (n_heads × d_head, d)  ← output projection

MLP / FFN:
  gate_proj:  (d, d_ff)           ← first projection
  up_proj:    (d, d_ff)           ← second projection (for SwiGLU)
  down_proj:  (d_ff, d)           ← output projection
```

### Common LoRA target configurations:

**Minimal (attention only — fastest):**
```python
target_modules=["q_proj", "v_proj"]
```

**Standard (attention + output):**
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Aggressive (all linear layers):**
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

**Why "all linear layers" often gives the best results:** The FFN/MLP layers contain most of the model's "knowledge" (factual information). Targeting only attention may miss these.

For VLMs, you can also target the projector:
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "multi_modal_projector.linear_1",
                "multi_modal_projector.linear_2"]
```

---

## 8. Dropout in LoRA

```python
lora_dropout=0.05
```

A small dropout applied to the LoRA matrices during training. Reduces overfitting when training on small datasets (<10K examples).

For large datasets or when using DPO/RLHF, dropout can be set to 0.

---

## 9. Memory Savings

### Why LoRA saves memory

```
Full fine-tuning: ALL 7B parameters have gradients → 14GB for gradients alone
LoRA: ONLY LoRA parameters (say, 20M) have gradients → 0.04GB for gradients

Base model parameters: stored in float16/bfloat16 (frozen, no gradients needed)
LoRA parameters: stored in float32 (small, needs full precision for training)
```

AdamW optimizer states only needed for LoRA parameters:
```
Full fine-tuning AdamW: 7B × 2 × 4 bytes = 56 GB (m + v)
LoRA AdamW:            20M × 2 × 4 bytes = 0.16 GB
```

---

## 10. QLoRA: LoRA + Quantized Base Model

**QLoRA (Dettmers et al., 2023)** combines LoRA with 4-bit NF4 quantization:

```
Standard LoRA:
  Base weights: bf16 (2 bytes/param) → 14 GB for 7B
  LoRA weights: fp32 (4 bytes/param) → ~0.1 GB

QLoRA:
  Base weights: NF4 4-bit quantized → 3.5 GB for 7B (4× compression)
  LoRA weights: bf16 (NOT quantized) → ~0.05 GB
  → Total: ~4 GB for 7B base + adapters
```

**NF4 (Normal Float 4):** A non-uniform 4-bit quantization optimized for normally-distributed weights. Better than naive INT4.

**Double quantization:** Quantize the quantization constants themselves, saving ~0.5GB.

### QLoRA workflow:
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # computation in bf16
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Trade-off:** QLoRA reduces quality vs LoRA in bf16 due to quantization noise.
- Use QLoRA when: single GPU, memory-constrained, you can accept ~1% accuracy loss
- Use LoRA in bf16 when: multi-GPU available, quality is critical

---

## 11. Merging LoRA Adapters

After training, LoRA can be merged back into the base model:
```python
model = model.merge_and_unload()
# W_merged = W_orig + (α/r) × B × A
# Now model has NO separate LoRA matrices
# Inference is identical to base model — no LoRA overhead
```

**Why merge?**
- Eliminates LoRA overhead at inference time
- Enables deployment like a normal model
- Required for quantization of the fine-tuned model

**When NOT to merge:**
- If you want to swap adapters dynamically (e.g., different LoRA per user)
- During training (you can't merge and train simultaneously)

---

## 12. LoRA vs Full Fine-Tuning — Full Comparison

| Aspect | Full Fine-Tuning | LoRA |
|--------|-----------------|------|
| VRAM (7B) | ~112 GB | ~20 GB |
| Trainable params | 7,000M | 5-50M |
| Training speed | Baseline | ~20% faster (less gradients) |
| Quality | Best (ceiling) | 95-99% of full FT quality |
| Catastrophic forgetting | High risk | Low risk (base weights frozen) |
| Multiple tasks | Need separate model | Multiple cheap adapters |
| Checkpoint size | 7B params (14 GB) | ~50M params (100 MB) |
| Use case | When you have lots of data + compute | Standard production case |

---

## 13. Complete VLM LoRA Fine-Tuning Code

```python
# Full LoRA fine-tuning example for Qwen2-VL-2B
# Requires: pip install transformers trl peft bitsandbytes pillow

import torch
from datasets import load_dataset, Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

# ─── 1. Load base model (4-bit QLoRA) ───────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# ─── 2. LoRA Configuration ───────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                          # rank: balance quality vs memory
    lora_alpha=32,                 # alpha: 2×r for 2× effective lr
    target_modules=[               # which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,             # small regularization
    bias="none",                   # don't adapt bias terms (small benefit)
    task_type=TaskType.CAUSAL_LM,  # decoder-only generation task
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20,971,520 || all params: 2,228,224,000
#         trainable%: 0.94%

# ─── 3. Dataset preparation ─────────────────────────────────────────────────
def format_data(sample):
    """Convert raw sample to Qwen2-VL chat format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},  # PIL Image or path
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
    }

def collate_fn(examples):
    """Batch processing: tokenize + process images."""
    texts = [
        processor.apply_chat_template(ex["messages"], tokenize=False,
                                       add_generation_prompt=False)
        for ex in examples
    ]
    image_inputs = [process_vision_info(ex["messages"])[0] for ex in examples]

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    # Loss only on assistant tokens (labels = -100 for non-assistant)
    labels = batch["input_ids"].clone()
    # Find assistant response start and mask everything before it
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # (Full masking logic depends on the specific template)
    batch["labels"] = labels
    return batch

# ─── 4. Training Configuration ───────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./outputs/vlm-lora",
    num_train_epochs=2,
    per_device_train_batch_size=1,          # small because images are large
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,          # effective batch = 1 × 8 = 8
    gradient_checkpointing=True,            # trade compute for memory
    optim="adamw_torch_fused",
    learning_rate=2e-4,                     # higher than full FT (LoRA convention)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,                              # use bfloat16 throughout
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,                     # keep only last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,            # important for VLMs
    dataloader_num_workers=4,
)

# ─── 5. Train ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

trainer.train()

# ─── 6. Save ─────────────────────────────────────────────────────────────────
trainer.save_model("./outputs/vlm-lora/final")
# Saves: adapter_config.json, adapter_model.safetensors (~100MB)
# Does NOT save base model weights (they're unchanged)

# ─── 7. Inference with trained adapter ───────────────────────────────────────
from peft import PeftModel

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./outputs/vlm-lora/final")

# Optionally merge for faster inference
model = model.merge_and_unload()
```

---

## 14. Checkpointing Strategy

```python
# During training, save both the full training state AND just the adapter
training_args = SFTConfig(
    save_steps=100,
    save_total_limit=3,                  # keep 3 checkpoints (rolling)
    load_best_model_at_end=True,         # restore best checkpoint at end
)

# What gets saved per checkpoint:
# checkpoints/checkpoint-100/
#   adapter_config.json         ← LoRA config
#   adapter_model.safetensors   ← LoRA weights only
#   optimizer.pt                ← AdamW state (for exact resume)
#   scheduler.pt                ← LR scheduler state
#   rng_state.pth               ← random number generator state
#   trainer_state.json          ← step, epoch, best metric
```

For exact training resume after GPU failure: you need ALL of these, not just the model weights.

---

## 15. Repositories to Study

```
Official PEFT:
  https://github.com/huggingface/peft
  Key file: src/peft/tuners/lora/layer.py
  What to look at: LoraLayer class, forward method
  Experiment: print(model) after get_peft_model() to see which layers have LoRA

TRL VLM examples:
  https://github.com/huggingface/trl
  Key file: examples/scripts/sft_vlm.py
  What to run: python sft_vlm.py --model_name_or_path Qwen/Qwen2-VL-2B-Instruct

LoRA paper (Hu et al.):
  https://arxiv.org/abs/2106.09685
  Read: Section 4 (practical experiments) and Table 6 (rank ablation)

QLoRA paper (Dettmers et al.):
  https://arxiv.org/abs/2305.14314
  Read: Section 2 (NF4 quantization) and Table 2 (quality comparison)
```

---

## 16. Interview Questions

**Q1 (Beginner): Why LoRA instead of full fine-tuning?**
Strong: "Full fine-tuning a 7B model requires ~112GB VRAM due to gradients and Adam optimizer states — that's more than the largest single GPU. LoRA adds low-rank matrices (A and B, typically 20M parameters total) only to specific weight matrices, leaving base model weights frozen. Only these 20M parameters need gradients and optimizer states, reducing VRAM from 112GB to ~20GB while achieving 95-99% of full fine-tuning quality."

Weak: "LoRA is faster and cheaper." (Too vague — shows no understanding)

Follow-up: "Where do the savings actually come from mathematically?"

**Q2 (Intermediate): How do you choose LoRA rank?**
Strong: "Start with r=16. This gives enough capacity for most domain adaptation tasks. If eval loss plateaus too high or the model fails on nuanced aspects of the task, double to r=32. If memory is constrained, drop to r=8. For tasks with fundamentally different output format (e.g., generating structured JSON vs natural prose), r=32 or higher. For minor style adaptation on already-capable models, r=8 is sufficient. Always run a rank ablation if compute allows: train r=8, r=16, r=32 for 1 epoch each and compare eval loss."

**Q3 (Advanced): In a VLM, should you apply LoRA to the vision encoder?**
Strong: "Rarely. The vision encoder is pre-trained on hundreds of millions of images — it already has excellent visual representations. Fine-tuning it risks degrading general visual understanding for marginal gains on your specific domain. The standard approach is: freeze the vision encoder, train LoRA on the LLM. However, if your domain has images with fundamentally different statistics (e.g., medical imaging, satellite imagery) that differ significantly from pre-training data, a small LoRA on the vision encoder's last few layers may help. Even then, use a lower rank and lower learning rate for the encoder than for the LLM."

---

## MUST KNOW Summary
- ΔW = B × A, where A: (r × d_in), B: (d_out × r), B initialized to zero
- Memory savings come from: frozen base weights = no gradients; tiny LoRA weights = tiny optimizer states
- B=0 at init so ΔW=0, preserving pre-trained capabilities at step 0
- QLoRA = 4-bit quantized base + LoRA in bf16 (~4GB for 7B)
- target_modules: at minimum q_proj + v_proj; ideally all linear layers
- Merge before deployment: model.merge_and_unload()
- Checkpoint = adapter weights + optimizer state + scheduler + RNG state
