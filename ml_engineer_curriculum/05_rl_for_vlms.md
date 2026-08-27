# TOPIC 05: Reinforcement Learning for VLMs

---

## WHY

DPO learns from fixed comparison data. RL learns from experience — the model generates responses, gets scored, and updates based on what works. This allows the model to discover strategies that weren't in the training data.

For verifiable tasks (math, code, OCR, visual QA), RL is often more powerful than DPO because you can provide exact binary rewards (correct/wrong) rather than imprecise human comparisons.

---

## 1. Core RL Concepts (applied to LLMs/VLMs)

### Policy (π)
The model itself. It's a function that takes a prompt (state) and generates a response (action).
```
π_θ(y | x) = probability of generating response y given prompt x
```
"Policy" = the current model weights.

### Reward (r)
A score given to a (prompt, response) pair. Can be:
- A trained reward model output: `r = rm(x, y) ∈ ℝ`
- A rule-based function: `r = 1 if answer_correct else 0`
- A composite: `r = α × accuracy + β × format + γ × length_penalty`

### Reference Model (π_ref)
A FROZEN copy of the model (usually the SFT checkpoint). Used to measure how far the policy has drifted.

### KL Divergence Penalty
```
KL(π_θ || π_ref) = E_y[log(π_θ(y|x) / π_ref(y|x))]
```
If the policy generates very different responses from the reference, KL is large.
This penalty prevents reward hacking: the model can't game the reward by generating gibberish.

### Rollout
The process of sampling responses from the current policy for a batch of prompts.

### Advantage (A)
How much better than average a particular response is:
```
A = r - baseline    (baseline ≈ average reward)
A > 0: this response is better than average → reinforce it
A < 0: this response is worse than average → suppress it
```

---

## 2. The RL Training Loop

```
┌─────────────────────────────────────────┐
│         For each training step:         │
│                                         │
│  1. SAMPLE prompts from dataset         │
│          ↓                              │
│  2. ROLLOUT: generate G responses       │
│     y₁, y₂, ..., yG ~ π_θ(y|x)         │
│     (current policy, temperature > 0)  │
│          ↓                              │
│  3. SCORE each response                 │
│     r_i = reward(x, y_i)               │
│          ↓                              │
│  4. COMPUTE ADVANTAGE                   │
│     A_i = (r_i - mean(r)) / std(r)     │
│     (group normalization for GRPO)      │
│          ↓                              │
│  5. COMPUTE POLICY LOSS                 │
│     L = -E[A_i × log π_θ(y_i|x)]       │
│       + β × KL(π_θ || π_ref)           │
│          ↓                              │
│  6. UPDATE POLICY                       │
│     θ ← θ - lr × ∇L                    │
│          ↓                              │
│  7. (Optional) Update reference model  │
│     every K steps (EMA update)          │
└─────────────────────────────────────────┘
```

---

## 3. Reward Hacking and KL Penalty

**Reward hacking** = the model finds responses that score high on the reward model but are NOT actually good.

**Example:**
- Task: generate creative image descriptions
- Reward model trained on verbosity as a proxy for quality
- Model learns to generate very long, repetitive descriptions
- High reward, terrible quality

**How KL penalty prevents this:**
```
Total objective = reward - β × KL(π || π_ref)
```
As the model diverges from the SFT model to game the reward, KL increases and subtracts from the objective. The model must balance reward maximization vs. staying close to the SFT baseline.

**Practical β values:**
- β = 0.01: loose, model can deviate significantly
- β = 0.1: standard, moderate constraint
- β = 0.5: tight, stays close to SFT

---

## 4. Mode Collapse

When all generated responses collapse to the same output (regardless of input). The model finds ONE high-reward pattern and only produces that.

**Symptoms:**
- All G rollouts are nearly identical
- Diversity score drops to ~0
- Training loss goes to 0 or NaN

**Prevention:**
- Temperature > 1.0 during rollouts
- KL penalty (high enough β)
- Diversity reward term: penalize if all G rollouts are identical
- Don't train for too many steps

---

## 5. GRPO Implementation

```python
from trl import GRPOConfig, GRPOTrainer

# Define the reward function
def compute_reward(prompts, responses, images=None):
    """
    For visual QA: reward = 1 if answer matches ground truth, else 0
    Returns: list of float rewards, one per (prompt, response) pair
    """
    rewards = []
    for prompt, response in zip(prompts, responses):
        ground_truth = extract_ground_truth(prompt)
        is_correct = normalize_answer(response) == normalize_answer(ground_truth)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards

# GRPO configuration
grpo_config = GRPOConfig(
    output_dir="./outputs/grpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,                  # very low — RL is unstable at high LR
    num_generations=8,                   # G: generate 8 responses per prompt
    temperature=0.8,                     # diversity in rollouts
    max_new_tokens=256,
    max_prompt_length=512,
    beta=0.1,                            # KL penalty
    bf16=True,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=sft_model,                     # policy (with LoRA)
    ref_model=ref_model,                 # frozen SFT model
    reward_funcs=compute_reward,         # your reward function
    args=grpo_config,
    train_dataset=dataset["train"],
)

trainer.train()
```

---

## 6. Reward Function Design for VLMs

This is the most critical and creative part of VLM RL.

### Verifiable rewards (cleanest)
```python
# OCR accuracy reward
def ocr_reward(image, generated_text, ground_truth_text):
    char_error_rate = edit_distance(generated_text, ground_truth_text) / len(ground_truth_text)
    return max(0.0, 1.0 - char_error_rate)

# Visual QA reward  
def vqa_reward(question, answer, ground_truths):
    # VQA allows multiple valid answers
    return 1.0 if answer.strip().lower() in [gt.lower() for gt in ground_truths] else 0.0

# Format compliance reward (response must be valid JSON)
def format_reward(response):
    try:
        json.loads(response)
        return 1.0
    except json.JSONDecodeError:
        return 0.0
```

### Soft rewards (for open-ended tasks)
```python
# CLIP-based visual relevance reward
def clip_reward(image, response):
    img_embedding = clip_model.encode_image(image)
    txt_embedding = clip_model.encode_text(response)
    similarity = cosine_similarity(img_embedding, txt_embedding)
    return float(similarity)

# Judge model reward
def judge_reward(image, prompt, response):
    judge_prompt = f"""Rate this response 1-5:
    Task: {prompt}
    Response: {response}
    Output only the number."""
    score = call_gpt4v(image, judge_prompt)
    return (float(score) - 1) / 4  # normalize to [0, 1]
```

### Composite reward
```python
def composite_reward(image, prompt, response, ground_truth):
    w1, w2, w3 = 0.5, 0.3, 0.2
    r1 = accuracy_reward(response, ground_truth)    # exact correctness
    r2 = clip_reward(image, response)               # visual relevance
    r3 = format_reward(response)                    # output format
    return w1 * r1 + w2 * r2 + w3 * r3
```

---

## 7. On-policy vs Offline Learning

| | On-policy (PPO, GRPO) | Offline (DPO) |
|--|----------------------|---------------|
| Data source | Fresh rollouts from current policy | Fixed precomputed dataset |
| Adapts to current policy | Yes | No |
| Compute per step | High (must generate responses) | Lower |
| Sample efficiency | Low (old rollouts are wasted) | High |
| Reward function | Can be dynamic/rule-based | Must be collected in advance |
| Distribution shift | Automatically corrected | Can become stale |
| Best for | Verifiable tasks (math, code, OCR) | Subjective preferences |

---

## 8. RLOO (REINFORCE Leave-One-Out)

A simpler alternative to GRPO that's also implemented in TRL:

```
For G responses y₁...yG per prompt x:
  r_i = reward(x, y_i)
  baseline_i = (1/(G-1)) × Σ_{j≠i} r_j     ← leave-one-out mean
  advantage_i = r_i - baseline_i
  
  L = -Σ_i advantage_i × log π_θ(y_i | x)
```

This is mathematically cleaner than GRPO (unbiased advantage estimator) but slightly higher variance.

```python
from trl import RLOOConfig, RLOOTrainer

config = RLOOConfig(
    rloo_k=4,             # K responses per prompt
    learning_rate=5e-7,
    beta=0.1,
    ...
)
trainer = RLOOTrainer(
    model=sft_model,
    ref_model=ref_model,
    reward_model=reward_model,
    args=config,
    train_dataset=dataset,
)
```

---

## 9. Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| NaN loss | LR too high or reward variance too large | Lower LR, normalize rewards |
| Mode collapse | KL too low or temperature too low | Increase β, increase temperature |
| Reward hacking | Reward model exploited | Add KL penalty, inspect samples |
| No learning | Reward variance too low | Make task harder, use contrastive pairs |
| Catastrophic forgetting | β too low | Increase β to 0.1-0.5 |
| OOM during rollouts | Too many generations (G too high) | Reduce G from 8 to 4 |

---

## 10. Interview Questions

**Q1 (Intermediate): What is the advantage in RL and how does GRPO compute it?**
Strong: "Advantage measures how much better a particular response is compared to a baseline. In standard RL, a learned value function estimates the baseline. GRPO avoids this by using group comparison: for each prompt, generate G responses (say, 8), compute rewards for all, then normalize: advantage_i = (r_i - mean(r)) / std(r). Responses above the group mean get positive advantage (reinforced), below get negative (suppressed). No separate value model is needed, which reduces memory and eliminates a source of training instability."

**Q2 (Advanced): Why does the KL penalty prevent reward hacking?**
Strong: "The training objective is: maximize expected reward MINUS β × KL(policy || reference). The reference is the frozen SFT model. If the policy deviates far from SFT to game a reward model (e.g., generating very long responses to exploit verbosity bias), the KL divergence from the reference increases. This subtracts from the total objective. The model must therefore balance reward maximization and staying close to SFT behavior. You tune β to set how tight this constraint is — higher β means the model can't deviate as much."

**Q3 (Advanced): When would you choose GRPO over DPO for VLM training?**
Strong: "GRPO when: (1) you have a verifiable reward signal (OCR accuracy, VQA correct/wrong, JSON format compliance) — these produce clean binary or continuous rewards without needing human comparison data. (2) You want the model to discover novel response strategies not in the training data. (3) The task requires multi-step reasoning where the model needs to try many approaches. DPO when: (1) you already have human preference data. (2) The task is subjective (creative quality, style) and hard to score automatically. (3) You want simpler, more stable training."

---

## MUST KNOW Summary
- Policy = the model being trained
- Reference model = frozen SFT checkpoint (measures divergence)
- KL penalty prevents reward hacking by constraining deviation from reference
- GRPO: generate G responses per prompt, normalize advantages within group
- Verifiable rewards (OCR accuracy, exact match) are cleanest for RL
- Mode collapse prevention: temperature > 0, KL penalty, diversity reward
- GRPO vs DPO: GRPO for verifiable tasks, DPO for subjective human preferences
