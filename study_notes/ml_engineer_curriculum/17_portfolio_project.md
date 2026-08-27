# TOPIC 17: Portfolio Project Design

---

## Project: AudioVision QA Evaluator

**Goal:** A system that evaluates the quality of AI-generated audio descriptions of visual scenes, using a fine-tuned VLM trained with preference learning and evaluated against frontier models.

This is deliberately designed to hit as many job requirements as possible while remaining buildable.

---

## Core Project (Complete This First)

### What it does
1. Takes a video clip
2. Extracts frames (visual) + audio (speech)
3. Transcribes speech with Whisper + forced alignment (word timestamps)
4. Uses a fine-tuned VLM (Qwen2-VL) to generate descriptions of visual scenes
5. Evaluates VLM description quality against human-annotated rubrics
6. Benchmarks against GPT-4V on the same task

### Why this covers the job
| Job requirement | How it's covered |
|-----------------|-----------------|
| VLM fine-tuning | Fine-tune Qwen2-VL on scene description data |
| LoRA | LoRA on the LLM backbone |
| Preference training | DPO on preference-annotated descriptions |
| Evaluation harness | Benchmark across 3 models with multiple metrics |
| Frontier benchmarking | Compare vs GPT-4V on same test set |
| ASR | Whisper for audio transcription |
| Forced alignment | WhisperX for word-level timestamps |
| Creative evaluation | Rubric for scene description quality |
| Multi-GPU training | 2-4 GPU setup on rented cluster |
| Checkpointing | Full fault-tolerant checkpoints |

---

## Core Project Implementation Plan

### Phase 1: Data Collection (Week 1)
```
Dataset: LLaVA-Instruct-150K (images + descriptions) for SFT
         + custom preference annotations for DPO

Preference data collection:
1. Use GPT-4V to generate 4 descriptions per image (varied quality)
2. Use Claude to rank them (1-4 quality score)
3. Form pairwise (chosen, rejected) DPO pairs from rankings
4. Validate 100 pairs with human judgment
   → IAA check
```

### Phase 2: SFT Fine-tuning (Week 2)
```python
# Target: Qwen2-VL-2B-Instruct (small, fast iteration)
# LoRA: r=16, alpha=32, all linear layers
# Dataset: 10K visual description examples
# Duration: ~2 hours on 2× A10 GPUs

from trl import SFTTrainer, SFTConfig
# ... (code from file 02)
```

### Phase 3: DPO Training (Week 2)
```python
# Take SFT checkpoint → run DPO
# Dataset: 2K (prompt, chosen, rejected) pairs
# β=0.1, 1 epoch

from trl import DPOTrainer, DPOConfig
# ... (code from file 04)
```

### Phase 4: Evaluation Harness (Week 3)
```python
# Models to evaluate:
models = {
    "baseline": "Qwen/Qwen2-VL-2B-Instruct",         # before SFT
    "sft": "./checkpoints/sft-final",                  # after SFT
    "dpo": "./checkpoints/dpo-final",                  # after DPO
    "gpt4v": "gpt-4o",                                 # frontier baseline
}

metrics = [
    bert_score,          # semantic similarity
    clip_similarity,     # visual grounding
    judge_model_score,   # GPT-4o rates descriptions 1-5
    rubric_score,        # your custom rubric score
]
```

### Phase 5: ASR + Forced Alignment Component (Week 3)
```python
# For video inputs with speech:
# 1. Extract audio track
# 2. Transcribe with Whisper
# 3. Align with WhisperX → word timestamps
# 4. Segment video based on speech pauses
# 5. Run VLM on each segment
```

### Phase 6: Production Serving (Week 4)
```bash
# Serve fine-tuned model via vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/dpo-merged \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9
```

---

## Extension Projects (Add These After Core)

### Extension A: GRPO with Verifiable Rewards
**Adds:** RL training, reward function engineering

```
Verifiable task: VQA on visual charts
Reward: exact match against known answer
Training: GRPO with 8 rollouts per prompt
Expected: 3-5% accuracy gain over DPO on chart QA
```

### Extension B: Multi-GPU Scale-up
**Adds:** Distributed training, FSDP

```
Upgrade to Qwen2-VL-7B (instead of 2B)
Train with 4× A100 using FSDP
Track: GPU utilization, throughput tokens/sec, cost per epoch
Expected: 2× more parameters → better quality; ~4× more cost
```

### Extension C: Production Cost Optimization
**Adds:** Quantization, inference optimization

```
Step 1: Quantize DPO model to AWQ INT4
Step 2: Compare:
  - bf16 throughput: X req/hour
  - AWQ INT4 throughput: Y req/hour
  - Quality loss: Z% on your eval set
Step 3: Serve AWQ model with vLLM
Step 4: Compute cost per 1000 requests at current GPU prices
```

---

## Completion Order

```
Week 1: Data collection + rubric + annotations
Week 2: SFT + DPO training
Week 3: Evaluation harness + frontier benchmark
Week 4: ASR + alignment integration + serving
Week 5+: Extensions (GRPO, 7B, cost optimization)
```

**Start with the smallest working version:**
- SmolVLM-2B (fits on CPU for debugging)
- 100 training samples (validate pipeline before scaling)
- 1 GPU (before multi-GPU)
- 1 metric (BERTScore) before full evaluation suite

---

## What to Showcase in Interviews

**The numbers you should know for your project:**
- "I fine-tuned Qwen2-VL-2B with LoRA r=16 on N examples for X hours on Y GPUs"
- "DPO improved BERTScore by Z% and judge model win rate by W%"
- "My 2B DPO model achieved 85% of GPT-4V's quality at 1/Xth the cost"
- "I used WhisperX for word-level timestamps with ~50ms precision"
- "Cohen's κ for my rubric was 0.73 (substantial agreement)"

**The story:**
"I built a system that evaluates scene descriptions from a fine-tuned VLM, comparing it against GPT-4V on a controlled benchmark. The most interesting part was building the creative quality rubric and measuring inter-annotator agreement to ensure the evaluation was actually meaningful."
