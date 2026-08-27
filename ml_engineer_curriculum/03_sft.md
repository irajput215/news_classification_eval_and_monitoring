# TOPIC 03: Supervised Fine-Tuning (SFT)

---

## WHY

SFT is the foundation. Before you do DPO or RL, the model must be SFT'd to follow instructions reliably. DPO on a poorly SFT'd model produces unpredictable results.

---

## 1. What SFT Does

SFT takes a pre-trained base model and teaches it to follow instructions by training on high-quality (instruction, response) demonstrations.

```
Base model:            "The capital of France" → " is a beautiful country..."
After SFT:             "What is the capital of France?" → "The capital of France is Paris."
```

Pre-training: predict next token on internet text (raw, unstructured)
SFT:          predict next token on instruction-following demonstrations (structured)

SFT teaches FORMAT, STYLE, and TASK BEHAVIOR. It does not fundamentally add new knowledge — it reveals knowledge the model already has in a useful format.

---

## 2. SFT Data Format

Every sample has three components:

```
System prompt (optional):
  "You are a helpful assistant that answers questions about images."

User instruction (required):
  [Image: image.jpg]
  "What is the main object in this image?"

Assistant response (required, this is what we train on):
  "The main object is a red bicycle leaning against a brick wall."
```

**Loss masking rule:**
```
System tokens:     IGNORE (no loss)
User tokens:       IGNORE (no loss)
Image tokens:      IGNORE (no loss)
Assistant tokens:  COMPUTE LOSS ← only this gets gradient
```

This is implemented with `labels = -100` for ignored tokens. PyTorch's CrossEntropyLoss ignores positions where label = -100.

---

## 3. SFT Data Quality vs Quantity

Quality matters more than quantity for SFT.

| Dataset size | Use case | Expected outcome |
|-------------|---------|-----------------|
| 100-500 | Extremely narrow task, specific domain | Only works on seen patterns |
| 1K-10K | Single task fine-tuning | Good task performance, limited generalization |
| 10K-100K | Multi-task fine-tuning | Good generalization within domain |
| 100K+ | General instruction tuning | Broad capability improvement |

**Curated 1K > Noisy 100K.** A dataset of 1,000 high-quality demonstrations consistently outperforms 100,000 noisy ones (LIMA paper finding).

---

## 4. SFT Training Hyperparameters for VLMs

```python
# Standard SFT hyperparameters
learning_rate = 2e-4        # Standard for LoRA SFT
                             # Lower (1e-4) for larger rank or full FT
num_epochs = 2-3             # More epochs = more overfitting risk
batch_size_effective = 8-32  # per_device × n_gpus × grad_accum
warmup_ratio = 0.03-0.1      # 3-10% warmup reduces initial instability
lr_scheduler = "cosine"      # cosine decay after warmup
weight_decay = 0.01          # small regularization
max_grad_norm = 1.0          # gradient clipping (prevents explosions)
```

---

## 5. Chat Templates

Different models expect different formats. ALWAYS use the model's chat template.

```python
# Qwen2-VL chat template applied automatically:
processor.apply_chat_template(messages, tokenize=False)
# Produces:
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# <|vision_start|><|image_pad|><|vision_end|>What is this?<|im_end|>
# <|im_start|>assistant
# This is a ...<|im_end|>
```

**Critical mistake:** Using the wrong chat template produces wrong loss masking and the model never learns.

---

## 6. Evaluating SFT Quality

During and after SFT, monitor:

1. **Training loss**: should decrease and plateau (not reach 0 — that's memorization)
2. **Eval loss**: should decrease then plateau; if it increases, you're overfitting
3. **Generation samples**: manually inspect 20-50 responses on eval set
4. **Task-specific metric**: accuracy, ROUGE, BERTScore depending on the task

**Early stopping signal:** eval_loss increases for 2+ consecutive evaluations → stop training.

---

## 7. Common SFT Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss doesn't decrease | Wrong loss masking (labels all -100) | Check label construction |
| Model outputs gibberish | Wrong chat template | Use processor.apply_chat_template |
| Model repeats input | Loss computed on input tokens too | Verify label masking |
| Catastrophic forgetting | LR too high, full fine-tuning | Use LoRA, reduce LR |
| Overfitting (eval loss rises) | Too many epochs, too small dataset | Reduce epochs, add dropout |
| OOM on first step | Images too large, batch too big | Reduce max_pixels, add grad checkpointing |

---

## 8. SFT → Next Steps

After SFT:
1. **DPO** — teach the model to prefer high-quality over low-quality responses
2. **RLHF** — optimize against a reward model using RL
3. **GRPO** — generate multiple responses, score them, update toward better ones

SFT alone produces good models. SFT + DPO produces great models. SFT + DPO + RL produces state-of-the-art models (this is roughly the recipe for Llama 3, Qwen2, etc.)

---

## Repository
```
TRL SFT documentation:
  https://huggingface.co/docs/trl/sft_trainer

LIMA paper (quality > quantity):
  https://arxiv.org/abs/2305.11206

VLM SFT example:
  https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py
```
