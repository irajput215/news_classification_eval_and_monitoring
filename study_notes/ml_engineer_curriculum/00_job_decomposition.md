# Job Decomposition — ML Engineer: Vision-Language Models

---

## The Role in Plain English

You are building and shipping an AI system that:
- Understands images AND audio AND text together
- Generates high-quality responses judged by creative craft standards
- Trains on large GPU clusters affordably
- Runs cheaply in production at scale
- Gets better over time via human preference and reinforcement signals

This is NOT a research role. It is an engineering role where quality is measurable, costs are tracked, and production reliability matters.

---

## Competency Decomposition Table

| Job Requirement | Knowledge Required | Practical Skill | Interview Importance |
|---|---|---|---|
| Fine-tune open VLMs | VLM architecture (encoder + projector + LLM), multimodal tokenization, image preprocessing | Load Qwen2-VL/LLaVA, apply LoRA, train with TRL | ★★★★★ |
| LoRA fine-tuning | Low-rank decomposition, rank/alpha/target modules, PEFT | Configure and train with HuggingFace PEFT | ★★★★★ |
| Preference training | RLHF, DPO, GRPO, reward modeling, preference datasets | Build preference dataset, run DPO with TRL | ★★★★★ |
| Reinforcement learning | Policy, reward, KL penalty, PPO, GRPO, advantage estimation | End-to-end GRPO training loop | ★★★★★ |
| Evaluation harness | Metrics (BLEU, BERTScore, CLIP, WER, win-rate), judge models, confidence intervals | Build reusable eval framework | ★★★★★ |
| Fair benchmarking | Controlled comparisons, leaderboard critique, statistical significance | Produce benchmark report with CIs | ★★★★ |
| Multi-GPU training | DDP, FSDP, ZeRO, NCCL, gradient sync, effective batch size | Launch multi-GPU jobs with torchrun/accelerate | ★★★★★ |
| GPU cluster economics | VRAM estimation, GPU comparison, cost calculation | Estimate cost per training run | ★★★★ |
| Checkpointing | Full training state (optimizer + scheduler + RNG), atomic writes, resume | Implement fault-tolerant training | ★★★★ |
| Production serving | vLLM/TGI/SGLang, continuous batching, KV cache, quantization | Deploy and load-test a VLM | ★★★★ |
| Low-cost inference | Quantization, batching, caching, throughput math | Calculate and reduce cost/request | ★★★★ |
| Audio preprocessing | Waveforms, mel spectrograms, VAD, chunking | Process audio for ASR models | ★★★ |
| ASR | Whisper, Wav2Vec2, CTC, WER/CER | Transcribe and evaluate | ★★★ |
| Forced alignment | CTC trellis, word/phoneme timestamps, MFA | Produce word-level timestamps | ★★★ |
| Creative → ML signals | Rubric design, annotation, IAA, reward models, preference datasets | Build annotation pipeline | ★★★★★ |

---

## MUST KNOW (Cannot go to interview without these)

### VLM Architecture
- What the vision encoder does (patch embeddings, ViT)
- What the projector/connector does
- How image tokens are inserted into the LLM's token sequence
- What is trainable vs frozen in different fine-tuning scenarios

### LoRA
- The low-rank decomposition: W_new = W_orig + (B × A) × α/r
- What rank and alpha control
- Which modules to target (q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj)
- Memory savings vs full fine-tuning
- QLoRA = LoRA + 4-bit quantized base model

### DPO
- Loss function: L_DPO = -log σ(β(log π_θ(y_w|x)/log π_ref(y_w|x) - log π_θ(y_l|x)/log π_ref(y_l|x)))
- Why it avoids needing a separate reward model
- Data format: (prompt, chosen, rejected) triples

### Multi-GPU Training
- effective_batch_size = per_device_batch × n_gpus × grad_accum
- ZeRO stages: 1 (optimizer states), 2 (+ gradients), 3 (+ parameters)
- FSDP vs DeepSpeed ZeRO conceptually

### Evaluation
- WER formula and calculation by hand
- When to use exact match vs BERTScore vs judge model
- Why bootstrap confidence intervals matter

### Production Serving
- KV cache: what it is and why it reduces latency
- Continuous batching vs static batching
- vLLM: what problem it solves

---

## SHOULD KNOW

- PPO vs GRPO vs RLOO trade-offs
- FSDP sharding strategies
- Gradient checkpointing vs gradient accumulation (different things!)
- Mel spectrogram construction from scratch
- CTC algorithm conceptually
- Bradley-Terry model for ranking
- Speculative decoding
- Quantization types: INT8, INT4, GPTQ, AWQ, GGUF

---

## NICE TO KNOW (After the above)

- Transducer/RNN-T ASR models
- IPO vs DPO objective differences
- ORPO (unified SFT + preference)
- Phoneme-level alignment with Montreal Forced Aligner
- NCCL internals (all-reduce ring algorithm)
- SGLang RadixAttention for KV reuse
- Flash Attention 2/3

---

## The Key Insight About This Role

Every part of this job requires you to make **quality measurable**. Whether you're:
- Designing a reward model (what does "good" look like numerically?)
- Building a benchmark (how do you know your model got better, not just luckier?)
- Monitoring drift (has production quality degraded since last week?)
- Training with preference data (what human judgment are you encoding?)

The core skill is: **taking a fuzzy human judgment and making it a number the model can learn from.**

---

## File Index

| # | File | Covers |
|---|------|--------|
| 01 | 01_vlm_architecture.md | VLM architecture, image preprocessing, multimodal tokenization |
| 02 | 02_lora_peft.md | LoRA from math to code, QLoRA, PEFT |
| 03 | 03_sft.md | Supervised fine-tuning for VLMs |
| 04 | 04_preference_training.md | RLHF, DPO, IPO, ORPO, GRPO — full comparison |
| 05 | 05_rl_for_vlms.md | PPO, GRPO, RLOO — theory + code |
| 06 | 06_evaluation_harness.md | Building evaluation pipelines, all metrics |
| 07 | 07_fair_benchmarking.md | Honest comparison against frontier models |
| 08 | 08_multi_gpu_training.md | DDP, FSDP, ZeRO, gradient sync |
| 09 | 09_gpu_clusters_and_cost.md | GPU selection, cost estimation, economics |
| 10 | 10_checkpointing.md | Full checkpoint state, fault tolerance |
| 11 | 11_production_serving.md | vLLM, TGI, SGLang, batching, quantization |
| 12 | 12_inference_cost.md | Cost math, throughput optimization |
| 13 | 13_audio_understanding.md | Waveforms, spectrograms, VAD, preprocessing |
| 14 | 14_asr.md | Whisper, Wav2Vec2, CTC, WER/CER |
| 15 | 15_forced_alignment.md | CTC trellis, word timestamps, MFA |
| 16 | 16_creative_evaluation.md | Rubrics, annotation, reward models, preference datasets |
| 17 | 17_portfolio_project.md | End-to-end project design |
| 18 | 18_interview_questions.md | 50+ questions with strong answers |
| 19 | 19_learning_curriculum.md | Phase-by-phase 15-phase curriculum |
