# TOPIC 16: Creative Craft → Measurable ML Signals

---

## WHY

This is the most unique and differentiating part of this job. Anyone can run a model. The ability to take "this dialogue sounds unnatural" and turn it into a training signal is what separates great ML engineers from commodity ones.

---

## 1. The Fundamental Problem

**Human creative judgment:**
"That scene felt emotionally flat. The pacing was wrong, and the ending line was weak."

This is:
- Subjective (different people might disagree)
- Multidimensional (multiple aspects: emotion, pacing, word choice)
- Contextual (depends on what came before)
- Hard to measure

**The ML engineer's job:**
Convert this into numbers a model can optimize against.

---

## 2. The Full Pipeline

```
Domain Expert or Human Judge
           ↓
    "What makes quality?"
           ↓
    Rubric Design
           ↓
    Annotation Guidelines
    (precise, unambiguous)
           ↓
    Annotator Training
           ↓
    Pilot annotation (50-100 samples)
           ↓
    Inter-Annotator Agreement check
    (Cohen's κ or Fleiss' κ)
           ↓
    Revise rubric if IAA < 0.7
           ↓
    Full annotation at scale
           ↓
    Labels / Preference pairs / Scalar ratings
           ↓
    Reward model training or DPO data creation
           ↓
    Automated metric for cheap, scalable evaluation
```

---

## 3. Rubric Design

A rubric converts vague quality into specific, answerable questions.

**Bad rubric:**
"Rate the dialogue quality 1-5."
(Too vague. Different annotators have different intuitions. Low IAA.)

**Good rubric:**
```
DIALOGUE NATURALNESS RUBRIC

Rate each dimension 1-4:

1. VOCABULARY APPROPRIATENESS
   1 = Uses words the character would never use given their background
   2 = Some vocabulary feels slightly off
   3 = Vocabulary is mostly appropriate with minor issues
   4 = Vocabulary is completely natural for this character and context

2. TURN-TAKING RHYTHM
   1 = Turns are all the same length; no variation
   2 = Variation exists but feels mechanical
   3 = Rhythm feels natural with occasional awkwardness
   4 = Rhythm feels like real conversation

3. SUBTEXT / WHAT GOES UNSAID
   1 = Everything stated explicitly; no subtext
   2 = Some subtext attempted but heavy-handed
   3 = Subtext present and mostly effective
   4 = Rich subtext that reveals character without stating it

4. EMOTIONAL CONSISTENCY
   1 = Emotional state shifts without motivation
   2 = Emotional arc exists but forced
   3 = Emotional arc is plausible, minor inconsistencies
   4 = Emotional arc is earned and consistent throughout
```

Each dimension is independently measurable and independently trainable.

---

## 4. Inter-Annotator Agreement (IAA)

**Cohen's Kappa (κ)** measures agreement beyond chance between 2 annotators:

```
κ = (P_observed - P_expected) / (1 - P_expected)

P_observed = fraction of items where annotators agree
P_expected = fraction expected to agree by chance alone

κ interpretation:
  < 0.0  = worse than chance (systematic disagreement)
  0.0-0.2 = slight agreement
  0.2-0.4 = fair agreement
  0.4-0.6 = moderate agreement
  0.6-0.8 = substantial agreement  ← minimum acceptable
  0.8-1.0 = near-perfect agreement ← ideal
```

```python
from sklearn.metrics import cohen_kappa_score

annotator_1 = [4, 3, 2, 4, 1, 3, 4, 2]
annotator_2 = [4, 3, 3, 4, 1, 2, 4, 2]

kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's κ = {kappa:.3f}")

# Weighted kappa for ordinal scales (penalizes large disagreements more)
kappa_weighted = cohen_kappa_score(annotator_1, annotator_2, weights="quadratic")
```

**Fleiss' Kappa:** Extends to K annotators instead of just 2.

**What to do when IAA is low:**
1. Review rubric — where do annotators disagree most?
2. Add examples to guidelines (positive and negative examples)
3. Run calibration sessions (annotators discuss cases together)
4. Simplify: binary choices (better/worse) have higher IAA than 5-point scales
5. If two experts fundamentally disagree: accept subjectivity, collect more annotators, use majority vote

---

## 5. Pairwise Preference vs Scalar Ratings

### Scalar ratings
```
Rate response quality: 1 | 2 | 3 | 4 | 5
```

Problems:
- Annotators use scale differently (one person's 3 is another's 4)
- "Magnitude estimation" is hard for humans
- Low IAA on absolute judgments

### Pairwise preference (recommended for DPO)
```
Given these two responses, which is better?
  Response A: "..."
  Response B: "..."
  → A is clearly better / A is slightly better / Tie / B is slightly better / B is clearly better
```

Why pairwise is better:
- Relative judgments are easier for humans than absolute
- Higher IAA (typically κ > 0.7 for pairwise vs κ < 0.5 for 5-point)
- Directly produces DPO training data format (chosen, rejected)

### Converting pairwise to DPO data:
```python
def pairwise_to_dpo_dataset(annotations):
    dpo_data = []
    for item in annotations:
        if item["preference"] in ["A_clearly", "A_slightly"]:
            dpo_data.append({
                "prompt": item["prompt"],
                "chosen": item["response_a"],
                "rejected": item["response_b"]
            })
        elif item["preference"] in ["B_clearly", "B_slightly"]:
            dpo_data.append({
                "prompt": item["prompt"],
                "chosen": item["response_b"],
                "rejected": item["response_a"]
            })
        # Skip ties
    return dpo_data
```

---

## 6. The Bradley-Terry Model

When you have many pairwise comparisons, how do you rank all items?

Bradley-Terry model learns a strength score for each item:
```
P(A > B) = strength_A / (strength_A + strength_B)
```

Training: maximize likelihood of observed pairwise outcomes.

```python
# Install: pip install choix
import choix
import numpy as np

# n_items: total number of responses being compared
# comparisons: list of (winner, loser) index pairs
n_items = 5
comparisons = [(0, 1), (0, 2), (1, 2), (3, 1), (4, 0), (4, 3)]

params = choix.ilsr_pairwise(n_items, comparisons)
# params: array of strength scores for each item
# Higher score = better quality

ranking = np.argsort(-params)  # descending order
print("Ranking:", ranking)
```

**Application:** Score all candidate responses for a prompt, not just best/worst. Useful when collecting 4-8 responses per prompt and wanting a full quality ranking.

---

## 7. Reward Models

A reward model is a fine-tuned LM that predicts scalar quality from (prompt, response):

```
Architecture: LLM + linear head → scalar output

Training data: (prompt, chosen, rejected) pairs
Training loss: -log σ(r_chosen - r_rejected)  ← maximize gap
```

### When to build a custom reward model:
- You have 1000+ labeled preference pairs
- Automated metrics don't correlate well with human judgment
- You need to score millions of generated responses cheaply

### When to use a judge model instead:
- < 1000 preference pairs (not enough for RM training)
- High diversity of prompts (RM might not generalize)
- You need nuanced explanations alongside scores

### Judge model vs reward model:
| | Judge Model (GPT-4 as judge) | Trained Reward Model |
|--|-----|------|
| Data needed | 0 (zero-shot) | 500+ labeled pairs |
| Cost | High (API cost per call) | Low (after training) |
| Speed | Slow (API latency) | Fast (local inference) |
| Consistency | Medium (slight variance) | High (deterministic) |
| Generalization | Excellent | Domain-specific |
| Best for | Small-scale eval, diverse tasks | High-volume scoring, RL reward |

---

## 8. Reward Hacking — The Creative Quality Version

**Scenario:** You train a reward model on "dialogue naturalness" ratings.
The RM learns that longer dialogues with more varied sentence lengths get higher scores.
Your GRPO-trained model learns to generate long dialogues with varied lengths.
Humans find these dialogues actually WORSE than before.

This is reward hacking. The RM was exploited.

**Mitigation:**
1. **Multi-dimensional rewards:** Don't use a single naturalness score. Use separate scores for vocabulary, rhythm, subtext, emotional consistency. Harder to hack all simultaneously.

2. **Adversarial testing:** Before using RM for training, explicitly try to "break" it. Generate deliberately bad outputs that you'd expect to get low scores, and intentionally game-proof outputs. Test if the RM gives correct scores.

3. **KL penalty:** Keep the trained model close to SFT model — limits how much it can drift toward gaming the reward.

4. **Periodic human spot-checks:** Sample 50 generated responses per week, have humans rate them. Compare human rating vs RM rating. When they diverge, RM has been hacked.

---

## 9. Translating Specific Creative Domains

### Written dialogue naturalness → ML signals
```python
def dialogue_naturalness_signals(dialogue: str, character_profile: dict) -> dict:
    """Extract measurable signals from dialogue."""
    sentences = sent_tokenize(dialogue)
    words = word_tokenize(dialogue.lower())
    
    return {
        # Vocabulary: does it match character education/background?
        "avg_word_length": np.mean([len(w) for w in words]),
        "unique_word_ratio": len(set(words)) / len(words),
        
        # Rhythm: sentence length variation
        "sentence_length_mean": np.mean([len(sent_tokenize(s)) for s in sentences]),
        "sentence_length_std": np.std([len(sent_tokenize(s)) for s in sentences]),
        
        # Contractions (casual speech marker)
        "contraction_rate": count_contractions(dialogue) / len(words),
        
        # Questions vs statements (conversational balance)
        "question_fraction": sum(1 for s in sentences if s.endswith("?")) / len(sentences),
        
        # Hedge words ("maybe", "I think", "probably") — uncertainty marker
        "hedge_word_rate": count_hedge_words(dialogue) / len(words),
    }
```

### Visual composition quality → ML signals
```python
def composition_signals(image) -> dict:
    """Measure visual composition quality signals."""
    import cv2
    img_array = np.array(image)
    
    return {
        # Rule of thirds: are key elements at intersection points?
        "thirds_score": compute_saliency_at_thirds(img_array),
        
        # Color harmony: are dominant colors harmonious?
        "color_harmony_score": compute_color_harmony(img_array),
        
        # Visual balance: is composition left-right balanced?
        "visual_balance": compute_spatial_balance(img_array),
        
        # Depth cues: is there clear foreground/background separation?
        "depth_score": estimate_depth_cues(img_array),
        
        # CLIP aesthetic score (LAION aesthetic predictor)
        "aesthetic_score": laion_aesthetic_predictor(image),
    }
```

### Voice performance quality → ML signals
```python
def voice_performance_signals(audio_path: str, script: str) -> dict:
    """Measure voice acting performance quality."""
    waveform, sr = librosa.load(audio_path, sr=16000)
    
    return {
        # Prosody: pitch variation (monotone = 0, expressive = high)
        "pitch_std": np.std(librosa.yin(waveform, fmin=50, fmax=500)),
        
        # Speech rate variability (robotic = low variance)
        "speech_rate_variation": compute_speaking_rate_variation(waveform, sr),
        
        # Pause naturalness (are pauses in semantically correct places?)
        "pause_naturalness": score_pause_positions(audio_path, script),
        
        # Energy dynamics (whisper vs shout; emotional range)
        "energy_dynamic_range": np.max(np.abs(waveform)) - np.percentile(np.abs(waveform), 25),
        
        # WER (did they say the script correctly?)
        "wer": compute_wer(transcribe(audio_path), script),
    }
```

---

## 10. Making Subjectivity Measurable Without Pretending It's Objective

**The honest position:**

Creative quality is partially subjective. You cannot perfectly measure it. What you CAN do:

1. **Decompose into dimensions** that are individually more objective
2. **Operationalize each dimension** with specific, answerable questions
3. **Calibrate annotators** to share a common standard
4. **Measure agreement** and only use dimensions with acceptable IAA
5. **Validate the metric** against held-out human judgments (not the training data)
6. **Report uncertainty** — always report variance, not just mean
7. **Acknowledge limitations** — your reward model captures some dimensions of quality, not all

**The mistake to avoid:** Treating your metric as ground truth. "Our reward model says this response is better" is not the same as "this response is actually better."

---

## 11. Interview Questions

**Q1 (Intermediate): How would you convert "this dialogue feels unnatural" into a training signal?**
Strong: "I'd decompose naturalness into specific dimensions: vocabulary appropriateness, turn-taking rhythm, use of contractions/hedges, and presence of subtext. Each dimension gets a 1-4 scale rubric with specific examples. I'd pilot with 50-100 samples using 2-3 annotators and measure Cohen's kappa per dimension. Dimensions with kappa < 0.6 get clearer guidelines or are dropped. I'd use pairwise preference (which of two dialogues is more natural) rather than absolute ratings — pairwise has higher IAA. The pairwise data becomes the (prompt, chosen, rejected) DPO dataset directly. I'd then train a reward model on 500+ pairs and validate it on a separate held-out set by checking correlation with fresh human ratings."

**Q2 (Advanced): Your reward model is giving high scores to verbose, repetitive responses. What's happening and how do you fix it?**
Strong: "This is reward hacking. My reward model learned that verbosity correlates with quality in my training data — probably because annotators associated detailed responses with effort and quality. The model being trained with GRPO is exploiting this spurious correlation. Fixes: First, add a length-normalized reward component: quality_score / log(response_length). Second, add a reward for conciseness when the task doesn't require length. Third, audit the reward model — sample 20 cases where it gives high scores to long responses; if the content is actually poor, retrain RM on a dataset where length is controlled. Fourth, add a KL penalty to limit drift. Fifth, move to multi-dimensional reward: score vocabulary, rhythm, and subtext separately — harder to hack all three simultaneously."

---

## MUST KNOW Summary
- Rubric: specific, answerable dimensions (not "rate quality 1-5")
- IAA: Cohen's kappa; < 0.6 = revise rubric; target > 0.7
- Pairwise preference has higher IAA than absolute ratings → better for DPO
- Bradley-Terry model: convert pairwise comparisons to global ranking
- Reward hacking: model finds spurious correlations; test adversarially before using RM for training
- Judge model (GPT-4): high quality, expensive, slow → for evaluation
- Trained reward model: cheap, fast, domain-specific → for RL reward
- Never treat metric as ground truth — validate against independent human judgments
