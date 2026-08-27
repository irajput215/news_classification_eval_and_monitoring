# TOPIC 04: Preference Training — RLHF, DPO, and Beyond

---

## WHY

SFT teaches the model to imitate demonstrations. But imitation is limited:
- You can't demonstrate every possible situation
- Human annotators vary in quality
- The model learns to mimic surface features, not underlying quality

Preference training teaches the model to make BETTER choices by learning from human comparisons. It is the core of alignment.

---

## 1. The Core Idea

```
Given a prompt:
  "Write a compelling opening line for a thriller novel."

Two responses:
  Response A: "It was a dark and stormy night."
  Response B: "She found the note tucked beneath her windshield wiper at 2 AM, 
               and her blood ran cold."

Human says: B is better.

Preference dataset entry: (prompt, chosen=B, rejected=A)

Now we train the model to:
  - Increase probability of generating B-like responses
  - Decrease probability of generating A-like responses
```

The data format: **(prompt, chosen, rejected)** triples.

---

## 2. The Full RLHF Pipeline (Classic Approach)

```
STEP 1: Supervised Fine-Tuning (SFT)
  Base model → SFT → SFT model (instruction follower)

STEP 2: Collect Preference Data
  SFT model generates 4-8 responses per prompt
  Human annotators rank/compare responses
  → Preference dataset: (prompt, chosen, rejected) pairs

STEP 3: Train Reward Model (RM)
  RM = LM with a scalar head instead of vocabulary head
  Training: maximize gap between reward(chosen) and reward(rejected)
  Loss: L_RM = -log σ(r(chosen) - r(rejected))

STEP 4: RL Fine-tuning with PPO
  Use PPO to maximize expected reward while staying close to SFT model
  Penalty: KL divergence from SFT model (prevents "reward hacking")
```

This pipeline is complex: you need 3 models simultaneously during PPO (policy, reference, reward model), significant compute, and it's notoriously unstable.

---

## 3. Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) is the key insight that changed the field:

> **You can optimize the policy directly from preference data without ever training a reward model.**

### The Math

DPO derives from RLHF but reformulates it as a classification problem.

**RLHF objective:**
```
maximize E[r(x, y)] - β × KL[π_θ(y|x) || π_ref(y|x)]
```

(maximize reward while staying close to reference model)

**DPO shows** this is equivalent to:
```
minimize -E[(x,y_w,y_l)] [log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

Where:
- y_w = chosen (winning) response
- y_l = rejected (losing) response
- β = temperature controlling deviation from reference (typically 0.1 to 0.5)
- π_θ = policy being trained
- π_ref = reference model (SFT model, frozen)

**In plain English:**
DPO increases the log-probability of chosen responses relative to reference AND decreases log-probability of rejected responses, controlled by how different each is from the reference.

### DPO Advantages
- **No reward model needed** — eliminates an entire training stage
- **More stable than PPO** — supervised loss, not RL
- **Less compute** — needs only policy + reference model (not reward model too)
- **Simpler to implement** — standard cross-entropy loss under the hood

### DPO Disadvantages
- **Offline** — learns from fixed dataset; cannot improve from new rollouts
- **Distribution shift** — if chosen/rejected responses are far from what the trained model generates, gradients are weak
- **β sensitivity** — wrong β → either collapse to reference model or diverge
- **Requires good SFT baseline** — bad SFT → bad DPO

### DPO Data Requirements
- SFT model (used as reference model, frozen)
- Preference dataset: `{prompt, chosen, rejected}` triples
- Both chosen and rejected must be plausible responses (not trivially bad)

### DPO Code (TRL)

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

# Dataset format expected by DPOTrainer:
# {
#   "prompt": "Write a compelling opening line...",
#   "chosen": "She found the note tucked beneath...",
#   "rejected": "It was a dark and stormy night."
# }
dataset = load_dataset("your/preference-dataset")

dpo_config = DPOConfig(
    beta=0.1,                           # KL penalty strength
    loss_type="sigmoid",                # standard DPO
    max_length=1024,
    max_prompt_length=512,
    output_dir="./outputs/dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,                 # much lower than SFT!
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
)

dpo_trainer = DPOTrainer(
    model=sft_model,                    # policy (trainable with LoRA)
    ref_model=ref_model,                # reference (frozen SFT model)
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

dpo_trainer.train()
```

---

## 4. IPO (Identity Preference Optimization)

IPO (Azar et al., 2024) is a small modification to DPO:

```
DPO loss: -log σ(β × Δ)
IPO loss:  (Δ - 1/(2β))²   where Δ = log ratio diff
```

**Why IPO?** DPO can become too "greedy" — it can overfit to the preference dataset by making log-ratios arbitrarily large. IPO has a bounded optimal solution, preventing this collapse.

**When to use:** When DPO overfits or when you have very clean, consistent preference data.

```python
dpo_config = DPOConfig(loss_type="ipo", beta=0.1)
```

---

## 5. ORPO (Odds Ratio Preference Optimization)

ORPO (Hong et al., 2024) unifies SFT and preference training into a single step:

```
L_ORPO = L_SFT + λ × L_OR

where L_OR = -log(sigmoid(log(odds_chosen) - log(odds_rejected)))
```

**Key advantage:** No need for a separate reference model. ORPO uses the SFT loss itself as the implicit regularizer.

**When to use:**
- Limited compute (no separate reference model run)
- You want SFT and alignment in one training pass
- Good starting point before SFT exists

**Limitation:** Less flexible than DPO; the SFT and preference signals can conflict.

```python
from trl import ORPOConfig, ORPOTrainer
config = ORPOConfig(lambda_=0.1, ...)  # λ controls preference strength
```

---

## 6. Reward Modeling

Even without PPO, reward models are useful for:
- Evaluating generated outputs
- Filtering datasets
- Scoring responses in GRPO/RLOO

### Architecture
```
LM backbone (same as policy) + scalar head
→ instead of predicting next token, predicts scalar reward r ∈ (-∞, +∞)
```

### Training
```python
# Bradley-Terry reward model loss
L = -log(σ(r_chosen - r_rejected))
# Maximize gap between chosen and rejected rewards
```

### Using a reward model (Llama-based example)
```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",   # a real reward model
    num_labels=1,
    torch_dtype=torch.bfloat16,
)

def get_reward(prompt, response):
    input_ids = tokenizer.encode(prompt + response, return_tensors="pt")
    with torch.no_grad():
        reward = reward_model(input_ids).logits.squeeze()
    return reward.item()
```

---

## 7. PPO (Proximal Policy Optimization)

PPO is the RL algorithm used in classic RLHF (ChatGPT, InstructGPT).

### The loop:
```
1. Rollout: Generate responses for a batch of prompts (using current policy)
2. Score:   Apply reward model to get rewards r
3. Compute advantage: A = r - baseline (how much better than average)
4. Update policy: gradient step to maximize A × log π(y|x)
5. KL penalty: penalize if new policy diverges too far from reference
6. Repeat
```

### PPO loss:
```
L_PPO = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)] - β × KL(π_θ || π_ref)

where r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)   (probability ratio)
      ε = 0.2 (clip range, keeps updates small)
```

### Why PPO is hard:
- Needs 3-4 models in memory simultaneously: policy, reference, reward, value
- Many hyperparameters (clip range, KL coefficient, GAE lambda, value coefficient)
- Susceptible to reward hacking
- On-policy: needs fresh rollouts constantly
- VRAM intensive

---

## 8. GRPO (Group Relative Policy Optimization)

GRPO (DeepSeek, 2024) is a simpler alternative to PPO used in DeepSeek-R1 and increasingly popular for VLMs.

### The key insight:
Instead of training a value model to estimate advantage, use GROUP COMPARISON:

```
For each prompt:
  Generate G=8 responses {y1, y2, ..., y8}
  Score each with reward function: {r1, r2, ..., r8}
  Normalize: A_i = (r_i - mean(r)) / std(r)   ← group-relative advantage
  Update policy toward high-advantage responses, away from low-advantage
```

No value model needed. Advantage is computed from the group itself.

### GRPO loss:
```
L_GRPO = -E[(1/G) Σ_i min(
    π_θ(y_i|x)/π_old(y_i|x) × A_i,
    clip(π_θ(y_i|x)/π_old(y_i|x), 1-ε, 1+ε) × A_i
)] + β × KL(π_θ || π_ref)
```

### Why GRPO is gaining popularity:
- No value model → ~25% less VRAM than PPO
- Stable training with interpretable hyperparameters
- Works well with verifiable rewards (math, code, OCR accuracy)
- Used in DeepSeek-R1, InternVL2, and other SOTA models

---

## 9. RLOO (REINFORCE Leave-One-Out)

RLOO is even simpler than GRPO — it's essentially REINFORCE with a clever baseline:

```
For G responses per prompt:
  baseline for response i = mean of rewards for all OTHER responses (leave-one-out)
  advantage_i = r_i - mean(r_j for j ≠ i)
```

This is an unbiased estimator of advantage with lower variance than plain REINFORCE.

RLOO is implemented in TRL and is a good starting point for RL training.

---

## 10. Comparison Table

| Method | Models needed | Data needed | Stability | Compute | Best for |
|--------|--------------|------------|-----------|---------|----------|
| SFT | Policy only | Demonstrations | High | Low | Starting point |
| DPO | Policy + frozen ref | (prompt, chosen, rejected) | High | Low-Medium | When SFT baseline exists |
| IPO | Policy + frozen ref | Same as DPO | Very High | Low-Medium | When DPO overfits |
| ORPO | Policy only | Same as DPO | High | Low | Single-stage SFT+align |
| RM | Reward model | Same as DPO | High | Medium | Input to PPO/GRPO |
| PPO | Policy + ref + RM + value | Generated rollouts | Low | Very High | When reward must be dynamic |
| GRPO | Policy + ref + reward fn | Generated rollouts | Medium-High | High | Math, code, verifiable tasks |
| RLOO | Policy + ref + reward fn | Generated rollouts | Medium | Medium-High | Simple RL starting point |

---

## 11. Preference Training for VLMs — Specific Challenges

### Challenge 1: Multimodal preference data is expensive
Text preference pairs are cheap to collect (one annotator reads two responses).
VLM preference pairs require showing an image + two responses — more time per annotation.

**Solution:** Use judge models (GPT-4V, Gemini) to generate pairwise preferences at scale.

### Challenge 2: Reward hacking through visual shortcuts
A model rewarded for "high quality descriptions" may learn to output verbose, generic descriptions that score well on the reward model but aren't actually helpful.

**Solution:** Mix multiple reward signals; test reward model on adversarial examples before using it.

### Challenge 3: Different visual tasks need different rewards
- Caption quality reward: should be measured by CLIP similarity + naturalness
- OCR reward: should be measured by character edit distance
- Visual QA reward: should be measured by accuracy (exact match or fuzzy match)
- Creative description: needs a judge model

**Multi-objective reward:**
```python
def visual_reward(image, prompt, response):
    ocr_score    = 1 - edit_distance(extracted_text, ground_truth) / len(ground_truth)
    caption_sim  = clip_similarity(image, response)
    fluency      = language_model_score(response)
    return 0.4 × ocr_score + 0.3 × caption_sim + 0.3 × fluency
```

---

## 12. DPO Data Collection Pipeline

```
1. Sample prompts from your domain
         ↓
2. Generate 4-8 responses with your SFT model
   (temperature 0.7-1.0 for diversity)
         ↓
3. Score responses (choose one):
   a. Human annotators: rank or choose preferred
   b. Judge model: GPT-4 / Gemini rates 1-5 or picks winner
   c. Automated metric: if task has ground truth
         ↓
4. Form pairs: best response = chosen, worst = rejected
   (or: any pair where one is clearly better)
         ↓
5. Quality filter: remove ties, near-ties, or low-confidence pairs
         ↓
6. Store: {prompt, chosen, rejected}
```

**Practical tip:** For VLMs, include the image path/URL in the prompt field. Make sure your DPOTrainer processes images correctly.

---

## 13. Interview Questions

**Q1 (Beginner): What is the difference between SFT and DPO?**
Strong: "SFT learns from demonstrations — it trains the model to imitate high-quality responses. DPO learns from comparisons — it trains the model to prefer chosen responses over rejected ones, without ever training a separate reward model. SFT teaches the model WHAT good responses look like. DPO teaches the model to CHOOSE better among alternatives. In practice, both are used sequentially: SFT first (to create a capable instruction-follower), then DPO (to align it toward human preferences)."

**Q2 (Intermediate): Why is DPO simpler than PPO-based RLHF?**
Strong: "PPO requires four models: policy, reference model, reward model, and value model. It uses on-policy rollouts, meaning you generate fresh responses at every training step. DPO requires only two models (policy and frozen reference) and trains on a fixed offline dataset. There are no reward model training stages, no sampling loops, and the loss is a standard supervised loss. The math shows that DPO implicitly learns the optimal policy that RLHF with PPO would find, but in a single stable training phase."

**Q3 (Advanced): How do you prevent DPO from overfitting to the preference dataset?**
Strong: "Three mechanisms: First, β (the temperature parameter, typically 0.1-0.5) controls how far the policy can deviate from the reference. Higher β = closer to reference = less overfitting. Second, keep training duration short — DPO often needs only 1-3 epochs. Third, use IPO instead of DPO if overfitting persists — IPO's loss is bounded and prevents log-ratios from growing arbitrarily. Fourth, add regularization: a small SFT component on the chosen responses (sometimes called DPO + SFT) prevents the model from degrading on non-preference-related tasks."

---

## MUST KNOW Summary
- Preference data format: (prompt, chosen, rejected) triples
- DPO = implicit reward maximization without training a reward model
- DPO loss: -log σ(β × (chosen log-ratio - rejected log-ratio))
- β controls deviation from reference (0.1 = close to reference)
- GRPO = group rollouts + group-normalized advantage (no value model)
- SFT → DPO is the standard alignment pipeline
- For VLMs: rewards must match the visual task (OCR ≠ caption ≠ creative)
