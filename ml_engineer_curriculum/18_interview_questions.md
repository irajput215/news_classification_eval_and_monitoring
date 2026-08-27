# TOPIC 18: 50+ Interview Questions — Full Reference

---

## SECTION 1: VLM Architecture

**Q1 (Beginner): What are the three main components of a Vision-Language Model?**
Intent: Tests understanding of VLM structure.
Strong: "Vision encoder (ViT), projector/connector (MLP), and language model backbone. The vision encoder converts an image into patch embeddings. The projector maps those to the LLM's embedding dimension. The LLM processes the combined visual + text token sequence and generates output."

**Q2 (Intermediate): Why are visual tokens inserted into the LLM's token sequence instead of passed through a separate pathway?**
Intent: Tests understanding of how VLMs achieve cross-modal attention.
Strong: "Self-attention in the LLM allows every text token to attend to every visual token. By inserting visual tokens into the sequence, text tokens can 'look at' the image naturally through the existing attention mechanism. A separate pathway would require a cross-attention module, adding parameters and complexity. The sequence insertion approach lets us reuse the pre-trained LLM unchanged and simply prepend visual tokens."

**Q3 (Advanced): What is training-serving skew in VLMs and how do you prevent it?**
Intent: Tests production ML awareness.
Strong: "Training-serving skew occurs when image preprocessing at inference differs from training. For example: training normalizes with ImageNet mean/std but serving uses a different range; or training resizes to 448px but serving passes 1080px. The model produces wrong predictions because it's processing different input distributions than it was trained on. Prevention: standardize preprocessing in a processor object, use the same processor at both training and inference, add preprocessing checks to the evaluation harness."

---

## SECTION 2: LoRA

**Q4 (Beginner): What does LoRA's B matrix being initialized to zero achieve?**
Intent: Tests understanding of LoRA initialization.
Strong: "At initialization, ΔW = B × A = 0 × A = 0. The model starts exactly at the pre-trained weights. This is critical because it means at step 0, the model produces the same outputs as the base model. If B were initialized randomly, you'd immediately corrupt the pre-trained representations before any training signal has been received, and the model would need many steps just to recover from initialization damage."

**Q5 (Intermediate): A colleague uses LoRA with r=64 and alpha=64, expecting better results than r=16. Explain when this is and isn't a good idea.**
Intent: Tests practical LoRA knowledge.
Strong: "r=64 gives 4× more trainable parameters than r=16 — better expressiveness for complex tasks but higher risk of overfitting on small datasets and more memory/compute. It's a good idea when: dataset is large (> 50K samples), the task requires significant behavioral change from the base model, compute allows it, and you've verified through ablation that r=16 underfits. It's NOT a good idea when: dataset is small (< 10K samples, risk of memorization), you need fast iteration, memory is constrained, or the base model already handles the task well and you only need minor style adjustment."

**Q6 (Advanced): You're training LoRA on a 7B VLM and want to target ALL linear layers. List them for a standard transformer model.**
Intent: Tests architecture knowledge.
Strong: "In the attention blocks: q_proj, k_proj, v_proj, o_proj. In the MLP/FFN blocks (SwiGLU activation): gate_proj, up_proj, down_proj. In the embedding layers: embed_tokens (optional, rarely trained). In the LM head: lm_head (optional). For VLMs, also optionally the projector: multi_modal_projector.linear_1, multi_modal_projector.linear_2. Targeting all linear layers gives the best results but increases trainable parameter count."

---

## SECTION 3: SFT

**Q7 (Intermediate): What is the loss mask in VLM SFT and what breaks without it?**
Intent: Tests SFT implementation details.
Strong: "The loss mask sets label=-100 for all tokens except the assistant's response. Without it, the model computes loss on image tokens and user instruction tokens — penalizing the model for 'not predicting' the input correctly. This creates wrong gradient signals: the model receives gradient toward reproducing its own visual inputs, leading to confused training. With correct masking, the model only learns to predict what comes AFTER the prompt — the assistant's response."

---

## SECTION 4: Preference Training

**Q8 (Beginner): What data format does DPO require?**
Intent: Tests basic DPO understanding.
Strong: "Three-column format: (prompt, chosen, rejected). Prompt is the input (can include image path for VLMs). Chosen is the preferred response. Rejected is the less preferred response. Both chosen and rejected should be plausible responses to the prompt — not one that's completely nonsensical (trivially easy) but one that a reasonable model might generate."

**Q9 (Intermediate): The DPO beta parameter — what happens with very high β vs very low β?**
Intent: Tests understanding of DPO regularization.
Strong: "β controls how far the trained model can deviate from the reference. Very low β (e.g., 0.01): the policy learns to strongly prefer chosen over rejected, potentially drifting far from SFT — risk of forgetting unrelated capabilities and reward hacking. Very high β (e.g., 1.0): the policy barely deviates from the reference model — alignment effect is negligible. Standard range: 0.1-0.5. Start with 0.1 and monitor: if eval loss on held-out SFT data increases significantly, β is too low."

**Q10 (Advanced): Compare GRPO vs DPO for OCR accuracy improvement on a VLM.**
Intent: Tests algorithm selection reasoning.
Strong: "GRPO is better here. OCR has a verifiable ground truth — the correct character sequence. You can define reward = 1 - CER (character error rate), which is exact and doesn't require human annotation. GRPO generates multiple transcriptions per image, scores them with CER, normalizes advantages within the group, and updates the policy. DPO requires collected human preference pairs — for OCR, this means having humans compare two transcriptions and choose the better one, which is less efficient and noisier than a deterministic CER comparison. For tasks with objective metrics (code, math, OCR), GRPO with a verifiable reward is the standard."

**Q11 (Advanced): How do you prevent DPO from degrading performance on tasks outside the preference dataset?**
Intent: Tests alignment degradation awareness.
Strong: "Three approaches: First, mix the DPO loss with SFT loss on the chosen responses: L = L_DPO + λ × L_SFT(chosen). This keeps the model's language modeling on chosen responses sharp. Second, monitor eval loss on a general held-out dataset (not just preference data) during training — stop if it increases. Third, use a high β (0.3-0.5) to constrain drift. Fourth, collect diverse preference data covering many domains, not just narrow task-specific examples. This reduces catastrophic forgetting of unrelated capabilities."

---

## SECTION 5: RL

**Q12 (Intermediate): What is the role of the reference model in GRPO/PPO?**
Intent: Tests RL for LLM understanding.
Strong: "The reference model is a frozen copy of the SFT model. It serves as a constraint: the KL term in the objective measures how far the current policy has drifted from the reference. Without it, the policy has no constraint and can drift to arbitrary behavior that scores high on the reward model but is meaningless or degenerate (reward hacking). The KL penalty forces the model to achieve high reward while staying close to the SFT baseline behavior. Typical: β=0.1 means moderate KL constraint."

**Q13 (Advanced): Your GRPO training shows mode collapse after 100 steps. How do you diagnose and fix it?**
Intent: Tests RL debugging.
Strong: "Diagnosis: print the G rollouts per prompt. If they're all near-identical (BLEU of rollouts vs each other > 0.9), that's mode collapse. Root cause: KL too low (model free to drift to one mode), temperature too low during rollout (greedy generation), or reward landscape has one dominant mode. Fixes: increase β to 0.3+; increase rollout temperature to 0.9-1.0; add diversity reward (penalize if rollout BLEU within group > 0.8); decrease learning rate; reduce G and increase prompt diversity."

---

## SECTION 6: Evaluation

**Q14 (Beginner): Give a worked WER calculation.**
Strong: "Reference: 'the quick brown fox'. Hypothesis: 'the quick red fox jumps'. 
  Align: the=✓, quick=✓, brown→red=S, fox=✓, (nothing)→jumps=I
  S=1, D=0, I=1, N=4 (reference words)
  WER = (1+0+1)/4 = 0.5 = 50%."

**Q15 (Intermediate): Why is BERTScore better than BLEU for evaluating VLM descriptions?**
Strong: "BLEU measures n-gram overlap. If the VLM generates 'a dog is sprinting across a green field' and reference is 'a canine running through grass', BLEU is near 0 despite equivalent meaning. BERTScore embeds both with a contextual model and computes cosine similarity in embedding space — 'dog' and 'canine', 'sprinting' and 'running' have high cosine similarity. For VLM evaluation where descriptions vary in phrasing, BERTScore correlates much better with human judgment."

**Q16 (Advanced): How do you know your benchmark is fair?**
Strong: "Five controls: (1) Same prompts — identical text with model-appropriate templates. (2) Same image input quality — don't give frontier models higher resolution. (3) Same generation parameters — temperature=0, same max_tokens. (4) Same test set — never overlap with training data; hash the test set. (5) Report CIs — with N=200 samples, 95% CI is meaningful; with N=20 it's not. Additionally: use a held-out test set you didn't use during development, report failure rates (API timeouts, refusals), and disclose any limitations (context length differences, system prompt differences)."

---

## SECTION 7: Multi-GPU Training

**Q17 (Beginner): What is the effective batch size formula?**
Strong: "effective_batch = per_device_batch × n_gpus × gradient_accumulation_steps. Example: per_device=2, 4 GPUs, accum=8 → effective batch = 2×4×8 = 64. This is important because training stability and optimal learning rate both depend on effective batch size, not per-device batch size."

**Q18 (Intermediate): What is ZeRO Stage 3 and when would you use it?**
Strong: "ZeRO Stage 3 shards everything across GPUs: model weights, gradients, and optimizer states. With N GPUs, each GPU holds 1/N of the model. This enables training models much larger than any single GPU's VRAM — 70B models on 8×80GB GPUs, for example. The cost: during forward/backward, each GPU must gather the layer shards it needs, causing more GPU-to-GPU communication. I'd use it when training models too large for per-GPU VRAM with ZeRO Stage 2 (weight sharding not needed)."

**Q19 (Advanced): During training on 4 GPUs, GPU utilization is 45%. What's the problem and how do you fix it?**
Strong: "45% utilization means the GPUs are idle more than half the time — almost certainly a CPU bottleneck. The GPUs finish a batch fast and wait for the next batch from the DataLoader. Fixes in order: First, increase DataLoader num_workers to 8-16. Second, add pin_memory=True to DataLoader (faster CPU→GPU transfer). Third, for VLMs with images: preprocess and cache image tensors to disk instead of processing on-the-fly. Fourth, increase per_device_batch_size to keep GPU busier per step. Fifth, add prefetch_factor=2 to DataLoader. If utilization reaches 85%+ after these, the bottleneck is solved."

---

## SECTION 8: GPU Clusters & Cost

**Q20 (Intermediate): How would you estimate training cost before starting?**
Strong: "Four steps: (1) VRAM requirement: LoRA on 7B bf16 ≈ 20GB → need A100 40GB or similar. (2) Throughput: benchmark 10 steps on target GPU, extrapolate tokens/second. (3) Time: total_tokens = n_samples × seq_len × n_epochs; time = total_tokens / (toks_per_sec × n_gpus). (4) Cost: hours × n_gpus × hourly_rate. Add 30% overhead for debugging runs and OOMs. Always check current provider pricing — GPU costs change frequently."

**Q21 (Advanced): How would you reduce training cost by 50%?**
Strong: "Five levers: (1) QLoRA instead of LoRA bf16 → reduce GPU requirement from A100 to A10, roughly proportional cost savings. (2) Spot instances → 60-80% cheaper, requires checkpointing every 50 steps + upload to S3. (3) torch.compile → 10-30% faster → proportionally cheaper. (4) Flash Attention 2 → 30-50% faster for long sequences. (5) Sequence packing → eliminate padding waste. QLoRA + spot + Flash Attention together can realistically achieve 60-70% cost reduction."

---

## SECTION 9: Checkpointing

**Q22 (Intermediate): How do you resume training after a GPU failure without restarting from scratch?**
Strong: "Three requirements: First, checkpoint frequency: save every 50-100 steps. Second, complete checkpoints: model weights + optimizer state + scheduler state + RNG state + step number. Third, durable storage: upload each checkpoint to S3/GCS immediately after saving (spot instances wipe local disk on termination). On restart: download latest checkpoint, restore all state, set DataLoader to skip already-processed batches by seeding with the correct RNG state. Maximum data loss: steps since last checkpoint."

**Q23 (Advanced): What is an atomic write and why does it matter for checkpoints?**
Strong: "An atomic write either fully succeeds or doesn't happen at all — there's no partial state. Write to a .tmp directory, then os.rename to final name. os.rename is atomic on most filesystems — the directory appears at its final name only when completely written. Without atomicity: if killed mid-write, you get checkpoint-100/ with model weights but no optimizer.pt. Next restart loads this 'checkpoint' but optimizer state is missing — training appears to resume but actually starts Adam from scratch, causing training instability that may not be obvious until you see the loss curve spike."

---

## SECTION 10: Production Serving

**Q24 (Beginner): What problem does continuous batching solve?**
Strong: "In static batching, you batch requests together and process them all until ALL finish. If request A generates 20 tokens and request B generates 200, the GPU is idle for 180 tokens' worth of compute waiting for B. Continuous batching inserts new requests the moment any request in the current batch finishes. This keeps every batch slot occupied, increasing GPU utilization from ~40-60% to > 80% and improving throughput."

**Q25 (Intermediate): How would you serve a 7B VLM cheaply?**
Strong: "Stack 4 optimizations: (1) AWQ INT4 quantization: 14GB model → 3.5GB, 1.8× more throughput. (2) Deploy on A10 24GB instead of A100 — INT4 model fits, much lower hourly cost. (3) vLLM with continuous batching and gpu_memory_utilization=0.9. (4) Request batching in the API layer: wait 50ms to collect concurrent requests before forwarding to vLLM. Together these reduce cost-per-request by 70-80% vs bf16 on A100."

**Q26 (Advanced): Explain PagedAttention and how it increases serving throughput.**
Strong: "KV cache stores attention keys and values for all previous tokens. Traditional allocation gives each request a contiguous chunk of GPU memory sized for its max possible context. When short requests finish, their memory is freed but can't be reused by requests of different sizes — fragmentation. PagedAttention manages KV cache in fixed-size pages (say, 16 tokens per page). Pages are allocated on demand and can be shared between requests that share the same prefix. This eliminates fragmentation, allowing 2-3× more concurrent requests in the same VRAM, directly translating to 2-3× higher throughput."

---

## SECTION 11: ASR & Audio

**Q27 (Intermediate): Calculate WER for: reference='the quick brown fox', hypothesis='a quick red fox'.**
Strong: "Align: 'the'→'a'=S, 'quick'=✓, 'brown'→'red'=S, 'fox'=✓. S=2, D=0, I=0, N=4. WER = 2/4 = 0.5 = 50%."

**Q28 (Advanced): Why is WER insufficient for evaluating creative voice performance?**
Strong: "WER measures whether the words were said correctly — it's binary at the word level. For creative voice performance, WER=0% (perfect transcript) is compatible with terrible performance: the actor could read every word correctly in a flat monotone with no emotional inflection. Creative quality requires measuring prosody (pitch variation, rhythm, pace), emotional expressiveness (energy dynamics, pause placement), and character authenticity. I'd use: pitch standard deviation (low = monotone), speech rate variation, pause naturalness (do pauses align with semantic boundaries?), and human ratings from trained listeners using a rubric."

---

## SECTION 12: Creative Evaluation

**Q29 (Intermediate): What is Cohen's Kappa and what threshold indicates acceptable inter-annotator agreement?**
Strong: "Cohen's kappa measures agreement beyond chance between two annotators. κ = (observed agreement - chance agreement) / (1 - chance agreement). Thresholds: < 0.4 = poor, 0.4-0.6 = moderate (need rubric improvement), 0.6-0.8 = substantial (acceptable for training), > 0.8 = near-perfect. For creative quality annotation, target > 0.6. If below, the rubric is ambiguous — add specific examples, simplify to binary (better/worse instead of 1-5), or break into more specific sub-dimensions."

**Q30 (Advanced): How do you build a reward model for creative dialogue quality?**
Strong: "Five steps: (1) Design rubric with 3-5 dimensions (vocabulary, rhythm, subtext, emotional arc), each with 1-4 scale and specific examples. (2) Collect 100 pilot annotations with 2-3 annotators; measure IAA; revise rubric until κ > 0.6. (3) Annotate 1000-2000 (dialogue, quality_score) pairs using pairwise format (which is better?) for higher IAA. (4) Train reward model: initialize from SFT LLM, add linear head, train with Bradley-Terry loss: -log σ(r_chosen - r_rejected). (5) Validate on 200 held-out pairs with fresh human ratings; target Spearman correlation > 0.7 between RM score and human score. If < 0.7, collect more diverse training data."

---

## SECTION 13: System Design

**Q31: Design a production pipeline for processing 10,000 video clips per day and generating quality descriptions.**
Strong outline:
```
1. Ingest: S3 upload triggers SQS queue
2. Pre-processing workers: extract frames (1 fps), extract audio (16kHz), VAD
3. ASR workers: Faster-Whisper on CPU cluster, batch processing, output word timestamps
4. VLM inference: vLLM with AWQ INT4, 2× A10 GPUs, continuous batching
5. Quality scoring: lightweight reward model or CLIP similarity
6. Output: descriptions + scores + timestamps stored in DynamoDB
7. Monitoring: request latency, GPU utilization, failure rates (Prometheus + Grafana)
8. Cost: estimate tokens per video → tokens per day → cost at current GPU price
```

**Q32: How would you scale from 1K to 1M inference requests per day?**
Strong: "1M/day = ~12 requests/second. Key changes: (1) Horizontal scaling: multiple vLLM replicas behind a load balancer. (2) Caching: if many requests share the same image (e.g., product photos), cache VLM predictions in Redis with content-hash key. (3) Model routing: classify request complexity; route simple text-only requests to Qwen2-1.5B, complex vision requests to Qwen2-VL-7B. (4) Quantization: AWQ INT4 → 1.8× more throughput. (5) Monitoring: auto-scale replicas on GPU utilization metric."

---

## Quick-Fire Questions (Know These Cold)

- What is the KV cache? → Stores attention K,V for previous tokens; O(1) per new token
- What is FSDP? → Shards model params + grads + optimizer across GPUs
- What is QLoRA? → 4-bit quantized base + LoRA adapters in bf16
- What is continuous batching? → Fill GPU slots immediately when requests complete
- What is speculative decoding? → Small draft model + large verifier = 2-4× speedup
- What is AWQ? → 4-bit quantization with activation-aware calibration
- What is DPO β? → Temperature controlling deviation from reference model
- What is CTC blank token? → Allow model to not output a character at frame t
- What is PSI threshold for drift? → 0.1 stable, 0.1-0.2 monitor, > 0.2 investigate
- What is Cohen's κ threshold? → > 0.6 acceptable, > 0.8 excellent
