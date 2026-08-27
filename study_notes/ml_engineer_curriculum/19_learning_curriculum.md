# TOPIC 19: 15-Phase Learning Curriculum

---

## How to Use This

Each phase has:
- What to read (file in this curriculum)
- What to implement (hands-on code)
- Repository to study
- Mastery criteria (how you know you're done)
- Interview questions unlocked

Complete phases sequentially. Do NOT skip to RL before finishing LoRA. Each phase builds on the previous.

---

## Dependency Graph

```
Phase 1: VLM Architecture
         ↓
Phase 2: LoRA / PEFT
         ↓
Phase 3: SFT
         ↓
Phase 4: Preference Training (DPO)
         ↓
Phase 5: RL (GRPO / RLOO)
         ↓
Phase 6: Evaluation Harness
         ↓
Phase 7: Fair Benchmarking
         ↓
Phase 8: Multi-GPU Training
         ↓
Phase 9: Checkpointing + Cost Optimization
         ↓
Phase 10: Production Serving
         ↓
Phase 11: Inference Cost Optimization
         ↓
Phase 12: Audio Understanding
         ↓
Phase 13: Forced Alignment
         ↓
Phase 14: Creative Evaluation
         ↓
Phase 15: End-to-End Project
```

---

## Phase 1: VLM Architecture

**Theory (File 01):**
- Vision encoder: ViT patch embeddings
- Projector: MLP dimension bridging
- Token sequence construction
- Trainable vs frozen components

**Implementation:**
```python
# TASK: Load Qwen2-VL-2B-Instruct, print the full architecture
# Count parameters in each component (vision encoder, projector, LLM)
# Process one image and print visual embedding shapes before and after projector

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
)
# Print model structure, count params per component
```

**Repository:**
```
https://github.com/haotian-liu/LLaVA
File: llava/model/llava_arch.py
Find: how visual tokens are prepended to text tokens
Find: where IGNORE_INDEX is used for loss masking
```

**Expected outcome:** You can draw the VLM architecture from memory and explain each component's role.

**Mastery criteria:**
- Can explain ViT patch embedding shape for a given image size
- Can name 3 projector types and their trade-offs
- Can explain why visual tokens are NOT included in the loss
- Can name 4 concrete VLMs and their distinguishing features

**Interview questions unlocked:** Q1, Q2, Q3 from File 18

---

## Phase 2: LoRA / PEFT

**Theory (File 02):**
- Low-rank decomposition: ΔW = B × A
- Rank, alpha, dropout, target modules
- B=0 initialization
- QLoRA vs LoRA

**Implementation:**
```python
# TASK: Fine-tune SmolVLM-256M with LoRA on 100 samples
# Use PEFT get_peft_model(), print trainable parameters
# Train for 10 steps, verify loss decreases
# Save adapter, reload, run inference

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~2M || all params: ~256M || trainable: ~0.8%
```

**Repository:**
```
https://github.com/huggingface/peft
File: src/peft/tuners/lora/layer.py
Class: LoraLayer → find the forward() method
Find: where B×A is computed and scaled by alpha/r
Find: where B is initialized to zeros
```

**Expected outcome:** Given a model architecture, you can choose appropriate LoRA config without looking it up.

**Mastery criteria:**
- Can derive memory savings for 7B model: full FT vs LoRA vs QLoRA
- Can explain rank ablation: when r=4 vs r=64 is appropriate
- Can implement LoRA config from scratch for any HF model
- Can explain merge_and_unload() purpose and when to use it

**Interview questions unlocked:** Q4, Q5, Q6

---

## Phase 3: SFT

**Theory (File 03):**
- Loss masking (assistant tokens only)
- Chat templates
- Dataset quality vs quantity
- Overfitting detection

**Implementation:**
```python
# TASK: Run TRL SFT example end-to-end
# Dataset: use a small VQA dataset (e.g., 500 samples from COCO-QA)
# Model: SmolVLM-500M
# Training: 2 epochs, verify eval loss curve looks correct
# Check: print a few generated samples to verify quality

# Run: python -m trl.scripts.sft_vlm --help
# Then run with SmolVLM-500M on your dataset
```

**Repository:**
```
https://github.com/huggingface/trl
File: examples/scripts/sft_vlm.py
Read every line and understand it before running
```

**Expected outcome:** You can fine-tune any HF VLM on a custom dataset.

**Mastery criteria:**
- Can explain why eval loss > train loss is expected (but rising eval loss means overfitting)
- Can identify the 3 most common SFT failure modes and their causes
- Can set up loss masking correctly for a new VLM's chat format

---

## Phase 4: Preference Training (DPO)

**Theory (File 04):**
- RLHF pipeline overview
- DPO loss derivation intuition
- β hyperparameter
- GRPO vs DPO use cases

**Implementation:**
```python
# TASK: Build a small DPO dataset and train
# 1. Generate 4 descriptions per image with SFT model (temp=0.8)
# 2. Score with CLIP similarity: best = chosen, worst = rejected
# 3. Run DPOTrainer with β=0.1 for 1 epoch
# 4. Compare: pre-DPO CLIP score vs post-DPO CLIP score on eval set

from trl import DPOTrainer, DPOConfig
dpo_config = DPOConfig(beta=0.1, loss_type="sigmoid", ...)
```

**Repository:**
```
https://github.com/huggingface/trl
File: trl/trainer/dpo_trainer.py
Find: the DPO loss computation
Find: where reference model log probs are computed
```

**Expected outcome:** You understand when DPO improves over SFT and when it doesn't.

**Mastery criteria:**
- Can explain DPO loss from first principles (no formula memorization required — understand the intuition)
- Can build a preference dataset from model outputs + automatic scoring
- Can monitor DPO training and detect overfitting
- Can compare DPO vs SFT on eval metrics

**Interview questions unlocked:** Q8, Q9, Q10, Q11

---

## Phase 5: RL for VLMs (GRPO / RLOO)

**Theory (File 05):**
- Policy, reward, reference model, KL penalty
- GRPO: group rollouts + group-normalized advantage
- Reward function design
- Failure modes: mode collapse, reward hacking

**Implementation:**
```python
# TASK: Run GRPO on a VQA task with exact-match reward
# Dataset: 500 VQA samples with known ground truth
# Reward: 1.0 if answer matches ground truth, 0.0 otherwise
# G=4 rollouts per prompt
# Monitor: reward distribution over training, mode collapse check

from trl import GRPOTrainer, GRPOConfig

def reward_fn(prompts, responses, **kwargs):
    return [1.0 if normalize(r) == normalize(gt) else 0.0 
            for r, gt in zip(responses, ground_truths)]
```

**Repository:**
```
https://github.com/huggingface/open-r1  (or trl GRPO examples)
https://github.com/QwenLM/Qwen2.5-VL — look at GRPO training scripts
```

**Expected outcome:** You can set up and monitor a GRPO training run.

**Mastery criteria:**
- Can explain group-relative advantage computation without notes
- Can design a reward function for 3 different VLM tasks
- Can detect and fix mode collapse
- Can choose between GRPO and DPO for a given task

**Interview questions unlocked:** Q12, Q13

---

## Phase 6: Evaluation Harness

**Theory (File 06):**
- Harness architecture: model → metrics → CI
- All metrics: EM, BLEU, ROUGE, BERTScore, CLIP, WER, win rate
- Bootstrap confidence intervals
- Why benchmarks mislead

**Implementation:**
```python
# TASK: Build a reusable EvalHarness class that:
# 1. Accepts any model_fn(prompt, image) → str
# 2. Computes 3 metrics (BLEU, BERTScore, CLIP)
# 3. Outputs 95% bootstrap CIs for each
# 4. Generates a markdown comparison table

# Run it on: baseline model, SFT model, DPO model
# Verify: DPO should improve CLIP score
```

**Repository:**
```
https://github.com/EleutherAI/lm-evaluation-harness
File: lm_eval/tasks/  ← how tasks are defined as YAML
File: lm_eval/evaluator.py ← main eval loop
```

**Expected outcome:** You have a reusable evaluation framework you can extend.

**Mastery criteria:**
- Can implement bootstrap CI from scratch (without scipy)
- Can choose the right metric for 5 different task types
- Can identify 3 ways a benchmark can mislead
- Can generate a professional benchmark report

**Interview questions unlocked:** Q14, Q15, Q16

---

## Phase 7: Fair Benchmarking

**Theory (File 07):**
- Controlled comparison checklist
- Leaderboard critique
- Frontier API setup (OpenAI, Anthropic)
- Benchmark report format

**Implementation:**
```python
# TASK: Run a 3-model benchmark
# Models: your DPO model, Qwen2-VL-2B baseline, GPT-4o-mini
# Dataset: 50 VQA samples (your held-out test set)
# Controls: same prompt, temp=0, same max_tokens
# Output: markdown table with mean + 95% CI + cost per request

# Compare: your model at GPU cost vs GPT-4o-mini at API cost
```

**Mastery criteria:**
- Can articulate 5 ways a benchmark comparison can be unfair
- Can produce a benchmark report with statistical significance
- Can explain the test set contamination problem
- Can design a private held-out test set

---

## Phase 8: Multi-GPU Training

**Theory (File 08):**
- DDP vs FSDP vs ZeRO
- effective_batch_size formula
- Gradient accumulation vs gradient checkpointing
- BF16 vs FP16

**Implementation:**
```bash
# TASK: Scale your SFT training from 1 GPU to 2 GPUs
# Method: use Accelerate with DDP
# Verify: loss curve matches single GPU (same effective batch size)
# Measure: tokens/second per GPU on 1 GPU vs 2 GPUs
# Expected: ~1.9× throughput (not perfect 2× due to communication)

accelerate config  # set up for 2 GPUs
accelerate launch train.py
```

**Repository:**
```
https://github.com/huggingface/accelerate
File: examples/complete_nlp_example.py
Examples for FSDP: examples/by_feature/fsdp_with_peak_mem_tracking.py
```

**Mastery criteria:**
- Can configure Accelerate for DDP and FSDP
- Can calculate effective batch size for any setup
- Can explain when to use FSDP vs DDP
- Can diagnose and fix low GPU utilization

**Interview questions unlocked:** Q17, Q18, Q19

---

## Phase 9: Checkpointing + Cost Optimization

**Theory (Files 09-10):**
- Full checkpoint contents
- Atomic writes
- Spot instance fault tolerance
- VRAM estimation

**Implementation:**
```python
# TASK: Implement fault-tolerant training loop
# Requirements:
# 1. Checkpoint every 50 steps (atomic write to .tmp then rename)
# 2. Upload to S3 after each checkpoint
# 3. On startup: check for latest checkpoint, resume if found
# 4. Register SIGTERM handler for spot instance preemption

import signal
def handle_sigterm(signum, frame):
    save_checkpoint(current_step, ...)
    upload_to_s3(...)
    sys.exit(0)
signal.signal(signal.SIGTERM, handle_sigterm)
```

**Mastery criteria:**
- Can explain what each component of a checkpoint stores and why
- Can implement atomic checkpoint saving from scratch
- Can estimate training cost for any model/dataset/GPU combination
- Can list 5 cost reduction strategies with expected savings

**Interview questions unlocked:** Q20, Q21, Q22, Q23

---

## Phase 10: Production Serving

**Theory (File 11):**
- KV cache: what it stores, memory cost
- Continuous batching
- vLLM, TGI, SGLang comparison
- Tensor parallelism, speculative decoding

**Implementation:**
```bash
# TASK: Deploy your DPO model with vLLM
# 1. Merge LoRA adapter: model.merge_and_unload(); save
# 2. Launch vLLM server
# 3. Write load test: send 100 concurrent requests
# 4. Measure: latency p50/p95, throughput (req/sec), GPU utilization

python -m vllm.entrypoints.openai.api_server \
    --model ./dpo-merged \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# Load test with locust or wrk
```

**Mastery criteria:**
- Can explain PagedAttention vs standard KV cache allocation
- Can deploy any HF model with vLLM
- Can measure and interpret serving latency/throughput metrics
- Can choose between vLLM/TGI/SGLang for different use cases

**Interview questions unlocked:** Q24, Q25, Q26

---

## Phase 11: Inference Cost Optimization

**Theory (File 12):**
- Cost formula: GPU cost / throughput
- AWQ quantization
- Model routing
- Output length control

**Implementation:**
```python
# TASK: Quantize your model to AWQ INT4
# 1. Quantize with autoawq
# 2. Measure quality: run eval harness on quantized model
# 3. Measure throughput: requests/second before vs after
# 4. Calculate: cost/request before vs after at current GPU price
# Document: quality-cost trade-off table
```

**Mastery criteria:**
- Can calculate cost per request from GPU hourly rate + throughput
- Can quantize any model to AWQ INT4
- Can explain quality-cost trade-off of different quantization levels
- Can design a request routing system

---

## Phase 12: Audio Understanding

**Theory (File 13):**
- Waveform, sample rate, amplitude
- STFT and mel spectrogram
- VAD
- Audio normalization and chunking

**Implementation:**
```python
# TASK: Build an audio preprocessing pipeline
# 1. Load a 3-minute audio file
# 2. Run VAD, extract speech segments
# 3. Compute mel spectrogram for one segment
# 4. Visualize with matplotlib: waveform + mel spectrogram side by side
# 5. Save speech-only segments as separate files

import librosa
import matplotlib.pyplot as plt
```

**Mastery criteria:**
- Can explain the mel spectrogram construction pipeline from waveform
- Can apply VAD to remove silence
- Can chunk audio with overlap for ASR
- Can explain why 16kHz is standard for speech

---

## Phase 13: ASR + Forced Alignment

**Theory (Files 14-15):**
- CTC loss and blank token
- Whisper encoder-decoder architecture
- WER/CER calculation
- CTC trellis for forced alignment
- WhisperX workflow

**Implementation:**
```python
# TASK: End-to-end audio pipeline
# 1. Transcribe a 10-minute podcast with Whisper large-v3
# 2. Get word-level timestamps with WhisperX
# 3. Calculate WER against a manual transcript
# 4. Run forced alignment on a known transcript with torchaudio
# 5. Export word timestamps as JSON

import whisperx
model = whisperx.load_model("large-v3", device="cuda")
result = model.transcribe("podcast.mp3")
# ... align and export
```

**Mastery criteria:**
- Can calculate WER by hand for any example (S+D+I)/N
- Can explain CTC blank token role in alignment
- Can produce word-level timestamps with WhisperX
- Can explain when to use MFA vs WhisperX

**Interview questions unlocked:** Q27, Q28

---

## Phase 14: Creative Evaluation

**Theory (File 16):**
- Rubric design
- Inter-annotator agreement (Cohen's κ)
- Pairwise preference vs scalar ratings
- Reward model training
- Reward hacking prevention

**Implementation:**
```python
# TASK: Build a creative quality evaluation pipeline
# 1. Design a 3-dimension rubric for "visual scene description quality"
# 2. Annotate 50 samples with your rubric
# 3. Have someone else annotate the same 50 samples
# 4. Calculate Cohen's kappa per dimension
# 5. Revise rubric for dimensions with κ < 0.6
# 6. Build a pairwise preference dataset from 100 samples
# 7. Train a small reward model (use DistilBERT + linear head)
# 8. Validate: Spearman correlation with human ratings on 20 held-out samples

from sklearn.metrics import cohen_kappa_score
```

**Mastery criteria:**
- Can design a rubric that achieves κ > 0.7 after iteration
- Can explain why pairwise has higher IAA than scalar ratings
- Can train a reward model and validate it independently
- Can identify reward hacking given training examples

**Interview questions unlocked:** Q29, Q30

---

## Phase 15: End-to-End Project

**Theory (File 17):**
- Project design
- Integration of all components
- What to showcase

**Implementation:**
Complete the AudioVision QA Evaluator project from File 17:
1. Data collection + rubric + pilot annotations
2. SFT fine-tuning (Qwen2-VL-2B, LoRA r=16)
3. DPO training on preference data
4. Evaluation harness: 3 models, 3 metrics, 95% CI
5. Frontier benchmark: your model vs GPT-4o-mini
6. WhisperX integration for audio clips
7. vLLM serving with cost tracking

**Final deliverables:**
- GitHub repo with clean, documented code
- Benchmark report: your model vs GPT-4V on controlled test set
- README with architecture diagram, training cost, key metrics
- A 3-minute recorded demo

**Mastery criteria:**
- Can give the 5-minute project walkthrough from File 13 (adapted for your project)
- Can answer all 32 interview questions in File 18
- Can describe your project's training cost to within 20%
- Can discuss 3 things you'd improve with more time/compute

---

## 30-Day Schedule

```
Week 1 (Days 1-7):
  Day 1-2: Phase 1 (VLM Architecture)
  Day 3-4: Phase 2 (LoRA)
  Day 5-6: Phase 3 (SFT) — run TRL SFT on SmolVLM
  Day 7:   Phase 4 theory (DPO)

Week 2 (Days 8-14):
  Day 8-9:  Phase 4 implementation (build DPO dataset + train)
  Day 10:   Phase 5 theory (GRPO)
  Day 11-12: Phase 5 implementation (GRPO with VQA reward)
  Day 13-14: Phase 6 (Evaluation harness — build it end-to-end)

Week 3 (Days 15-21):
  Day 15:   Phase 7 (Fair benchmarking — run 3-model comparison)
  Day 16-17: Phase 8 (Multi-GPU — scale to 2 GPUs)
  Day 18:   Phase 9 (Checkpointing + cost estimation)
  Day 19-20: Phase 10-11 (Serving + quantization)
  Day 21:   Phase 12-13 (Audio + ASR + alignment)

Week 4 (Days 22-30):
  Day 22:   Phase 14 (Creative evaluation + rubric)
  Day 23-30: Phase 15 (End-to-end project)
```

---

## Priority Matrix for Interview Prep

**If you have 1 week:** Phases 1, 2, 4, 6, 10 (VLM arch + LoRA + DPO + eval + serving)

**If you have 2 weeks:** Add phases 3, 5, 7, 8, 9 (SFT + GRPO + benchmarking + multi-GPU + checkpointing)

**If you have 1 month:** Complete all 15 phases

**Non-negotiable for this role:** Phases 1, 2, 4, 6, 16 (architecture, LoRA, DPO, eval, creative signals)

---

## MUST KNOW (Final Summary)

By the end of all 15 phases, you must know these cold:

**VLM:** ViT → projector → LLM token sequence, loss masking on assistant tokens only

**LoRA:** ΔW = (α/r) × B×A; B=0 init; r=16 standard; QLoRA = 4-bit base + LoRA

**DPO:** No reward model needed; β controls deviation from reference; (prompt, chosen, rejected) format

**GRPO:** G rollouts per prompt; advantage = (r - mean(r)) / std(r); no value model

**Evaluation:** BERTScore > BLEU for open-ended; always report 95% CI; control all variables vs frontier

**Multi-GPU:** effective_batch = per_device × n_gpus × grad_accum; FSDP for large models

**Serving:** KV cache eliminates O(n²); continuous batching → > 80% GPU util; vLLM is standard

**Cost:** cost/req = GPU_hourly / (req/hour); AWQ gives 1.8× throughput; spot = 60-80% cheaper

**ASR:** WER = (S+D+I)/N; Whisper = encoder-decoder; Wav2Vec2 = CTC

**Creative eval:** Rubric + IAA (κ > 0.6) + pairwise + reward model + validate on held-out human ratings
