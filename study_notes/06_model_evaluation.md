# TOPIC 06: Model Evaluation & Metrics

---

## 1. What is it?

Model evaluation is the quantified measurement of how well a trained model generalizes to unseen data. For classification tasks, this means computing specific metrics that capture different aspects of prediction quality.

There are TWO evaluation steps in this project:
1. **During training** (`train.py`): Evaluate on VALIDATION set → select best model
2. **After training** (`evaluate.py`): Evaluate Production model on TEST set → final reporting

These must be kept separate. Never use the test set for model selection.

---

## 2. Metrics Used in THIS Repository

### Implemented metrics (src/train.py lines 105-122):

```python
def compute_metrics(y_true, y_pred, y_prob, label_names):
    n_classes = len(label_names)
    average   = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1_macro"  : f1_score(y_true, y_pred, average="macro"),
        "precision" : precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, average=average, zero_division=0),
    }

    # AUC-ROC (multi-class uses OvR strategy)
    try:
        if n_classes == 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["auc_roc"] = 0.0

    return metrics
```

---

## 3. Every Metric Explained From First Principles

### 3.1 Confusion Matrix (the foundation)

For binary classification, with 4 outcomes:
```
                    Predicted Positive  Predicted Negative
Actual Positive:    TP (True Positive)  FN (False Negative)
Actual Negative:    FP (False Positive) TN (True Negative)
```

For AG News (4 classes), this extends to a 4×4 matrix where:
- Diagonal = correctly classified
- Off-diagonal = misclassified (which class was confused with which)

### 3.2 Accuracy

```
Accuracy = (TP + TN) / Total = Correct Predictions / All Predictions
```

**When it's useful:** Classes are balanced. AG News is balanced (4 classes × 2500 each).

**When it's misleading:** Imbalanced classes. If 95% of emails are not spam, a model that predicts "not spam" for everything has 95% accuracy but is useless.

**In this project:** Used as the PRIMARY selection metric for model comparison. Reasonable because AG News is balanced.

### 3.3 Precision

```
Precision = TP / (TP + FP)
= "Of all the times I predicted class X, how often was I right?"
```

**Example:** Model predicts 100 articles as "Sports". 90 are actually Sports, 10 are wrong.
- Precision = 90/100 = 0.90

**When to prioritize:** When false positives are costly. Email spam filter: a false positive (marking valid email as spam) is worse than a false negative.

### 3.4 Recall (Sensitivity)

```
Recall = TP / (TP + FN)
= "Of all actual class X articles, how many did I correctly find?"
```

**Example:** There are 200 actual Sports articles. Model finds 170 of them.
- Recall = 170/200 = 0.85

**When to prioritize:** When false negatives are costly. Medical diagnosis: missing a cancer case (false negative) is worse than a false alarm.

### 3.5 F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= Harmonic mean of precision and recall
```

**Why harmonic mean?** If precision = 1.0 and recall = 0.0 (useless model), arithmetic mean = 0.5 (looks OK). Harmonic mean = 0.0 (correctly shows uselessness).

**Macro F1:** Compute F1 for each class separately, then average equally. Fair to all classes regardless of size.

**This project uses `f1_macro`** — the right choice for multi-class balanced classification.

### 3.6 AUC-ROC

ROC = Receiver Operating Characteristic curve.
AUC = Area Under the ROC Curve.

**The ROC curve plots:**
- X-axis: False Positive Rate = FP/(FP+TN)
- Y-axis: True Positive Rate (= Recall) = TP/(TP+FN)
- At different decision thresholds (0.0 to 1.0)

**AUC interpretation:**
- 0.5 = random classifier (diagonal line)
- 1.0 = perfect classifier
- 0.9+ = excellent
- 0.7-0.9 = good
- < 0.7 = weak

**Multi-class OvR (One vs Rest):** For 4 classes, compute 4 binary AUC values (World vs rest, Sports vs rest, etc.) and average them.

**Why AUC-ROC matters:** It measures discriminative ability regardless of threshold. You might deploy with a high-confidence threshold — AUC tells you how good the underlying probabilities are.

### 3.7 When NOT to use AUC-ROC

**Use PR-AUC (Precision-Recall AUC) when:**
- Dataset is very imbalanced
- You care more about the positive class than overall classification

In AG News (balanced, 4 classes), AUC-ROC is fine. For fraud detection (0.1% fraud), use PR-AUC.

---

## 4. Multi-Class Averaging Strategies

The project uses `average="macro"` for F1, precision, recall (when multi-class):

| Strategy | Formula | When to use |
|----------|---------|-------------|
| `macro` | Average of per-class metric (unweighted) | Balanced classes, want equal treatment of all classes |
| `weighted` | Average weighted by class support (size) | Imbalanced classes, overall performance matters |
| `micro` | Aggregate TP/FP/FN across classes, then compute | When you want to weight by instance count |
| `binary` | For binary classification only | Only 2 classes |

**AG News has equal classes → macro is correct.**

---

## 5. Where is it used in THIS repository?

### During Training (src/train.py):
```python
# Validation metrics logged to MLflow
mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
# val_accuracy, val_f1_macro, val_precision, val_recall, val_auc_roc
```

### Final Evaluation (src/evaluate.py):
```python
# Loads Production model
model_uri = f"models:/{model_name}/Production"
pipeline = mlflow.sklearn.load_model(model_uri)

# Evaluates on test set
metrics, y_pred, y_prob = evaluate(pipeline, test, label_names)

# Saves:
# reports/final_classification_report.txt  → per-class metrics
# reports/final_metrics.json               → summary numbers
# reports/final_confusion_matrix.png       → visual matrix
```

---

## 6. The Classification Report

```
                  precision    recall  f1-score   support
     Business       0.91      0.90      0.90       375
      Sci/Tech       0.90      0.91      0.90       375
        Sports       0.98      0.97      0.97       375
         World       0.91      0.91      0.91       375

      accuracy                           0.92      1500
     macro avg       0.92      0.92      0.92      1500
  weighted avg       0.92      0.92      0.92      1500
```

**Reading this table:**
- Sports has 0.98 precision: when the model says "Sports", it's right 98% of the time
- Sports has 0.97 recall: 97% of all actual Sports articles are correctly found
- Support: number of actual instances in each class
- Macro avg: simple average across classes (what `f1_macro` reports)
- Weighted avg: weighted by support (here same as macro since classes are balanced)

**This tells you WHERE the model struggles:** If Business and Sci/Tech have lower scores than Sports and World, it means those two are being confused with each other (business news about tech companies, etc.)

---

## 7. The Confusion Matrix Plots

`save_confusion_matrix()` in train.py creates TWO plots side by side:
1. **Counts**: Raw number of samples in each cell
2. **Normalized**: Fraction (0-1) per row — shows per-class recall

**How to read:**
- Diagonal = correct predictions
- Row i, Column j = "actual class i, predicted as class j"
- A hot off-diagonal cell: model frequently confuses class i for class j

**Business vs Sci/Tech confusion** is the most common error in news classification — "Apple reports record revenue" could be Business OR Sci/Tech.

---

## 8. When Should a Model Be Retrained?

This is NOT a simple question. Do NOT say "retrain whenever drift occurs."

### Decision Framework:

```
1. Is there data drift?
   ↓ YES
2. Is the drifted feature important to the model?
   ↓ YES
3. Has model performance actually degraded?
   (measured on labeled recent data or via proxy metrics)
   ↓ YES
4. Is the degradation significant?
   (e.g., accuracy dropped from 0.92 to 0.85 — exceeds threshold)
   ↓ YES
5. Retrain with new data
```

Excellent question — this is exactly the gap that trips people up in interviews. Here's the honest answer:

---

## You Can't Measure Accuracy Without Labels

Accuracy requires knowing the **ground truth**. If you never have labels for new production data, you **cannot directly measure** accuracy drop. This is the core tension.

So production ML systems use a combination of **three strategies**:

---

## Strategy 1: Delayed Labels (Most Common)

In many real systems, **labels arrive eventually** — just with a delay.

```
User uploads news article → Model predicts "Sports"
                              ↓
         24 hours later: editors manually categorize it → "World"
                              ↓
         Compare: prediction vs delayed label → accuracy on recent data
```

**Examples:**
- Fraud detection: label = did the transaction turn out to be fraud? (confirmed in days/weeks)
- Loan default: label = did the user default? (confirmed in months)
- News classification: label = how did an editor actually categorize it?

**Approach:** Maintain a rolling window of labeled recent data (e.g., last 7 days of confirmed labels). Compute accuracy on this window. If it drops below threshold → trigger retraining.

---

## Strategy 2: Proxy Metrics (When Labels Don't Exist)

When you'll **never get labels**, you monitor signals that **correlate with** accuracy:

| Proxy Metric | What it detects | Limitation |
|---|---|---|
| Prediction distribution shift | Model suddenly predicts "Sports" 80% vs historical 25% | Doesn't tell you if the predictions are wrong |
| Confidence score distribution | Model confidence drops from avg 0.91 → 0.73 | Low confidence ≠ always wrong |
| Model input drift (PSI) | Input features changed significantly | Doesn't directly mean accuracy dropped |
| Business KPIs | Click-through rate drops on classified articles | Confounded by non-model factors |

**The honest limitation:** proxy metrics are *signals*, not proof. A model can have drifted inputs but still be correct. Or stable inputs but still be wrong.

---

## Strategy 3: Active Sampling / Human Spot Checks

Even without full labels, you can **sample a small percentage** of predictions and have humans verify them:

```
1,000 predictions/day
     ↓
Sample 50 (5%) randomly
     ↓
Human verifier checks: was the prediction correct?
     ↓
Estimated accuracy on recent data = correct / 50
     ↓
If this drops from 0.92 → 0.85 over 3 weeks → retrain
```

This is what companies like Google and LinkedIn do in production. They pay annotators to label a small ongoing sample.

---

## The Revised Decision Framework (Honest Version)

```
1. Detect: Is there input data drift? (PSI on features)
   ↓ YES
2. Is the drifted feature important? (feature importance check)
   ↓ YES — raise alert
3. Check proxy signal: Has prediction distribution shifted?
   ↓ YES — raise alert
4. Check actual performance:
     a. Do you have delayed labels? → compute accuracy on last N days
     b. No labels? → active sample: label 50-100 samples manually
     c. No budget for labeling? → use confidence score degradation as proxy
   ↓ If performance drop is confirmed and exceeds threshold
5. Retrain
```

---

## Key Interview Insight

If an interviewer asks "how do you know performance degraded without retraining?" — the correct answer is:

> "In production you need a **labeling strategy**. Either delayed labels that naturally arrive, active sampling where you pay for spot labels, or business KPIs as a proxy. Drift detection alone only tells you the input distribution changed — it doesn't tell you the model is wrong. Monitoring without any path to ground truth is incomplete."

The production reality is:
- **Small teams:** confidence score degradation + quarterly human spot check
- **Mature teams:** delayed labels + automated accuracy tracking
- **Large teams:** dedicated labeling pipelines with ongoing sampled annotation


### What to monitor before deciding:
- **Data quality metrics**: null rate, schema compliance (always monitor)
- **Input feature distributions**: PSI, KS test on text features (monitor regularly)
- **Prediction distribution**: Are model outputs shifting? (e.g., more Sports predicted than usual)
- **Actual performance**: If you have labels, compute metrics on recent data
- **Business metrics**: Click-through rate, user corrections, downstream KPIs

### Common retraining triggers:
- Performance drops below threshold (e.g., F1 drops > 5%)
- PSI > 0.2 AND performance proxy confirms degradation
- Scheduled retraining (weekly/monthly regardless of drift)
- New data class added to the problem space
- Business rule change (e.g., new news category "AI" added)

---

## 9. Production Perspective

**What this repo implements:**
- Validation metrics used for model selection
- Test metrics logged back to the MLflow run
- Classification report and confusion matrix saved as artifacts
- Accuracy threshold gate (0.88) for registration

**What a production system would add:**
- **Holdout test set locked away**: Not touched until final evaluation — this project uses it for evaluation but also potentially for CI data generation (blurred boundary)
- **Statistical significance testing**: Is model B significantly better than model A?
- **Cost-sensitive evaluation**: Some misclassifications cost more than others
- **Calibration metrics**: Are the probabilities well-calibrated? (Brier score)
- **Fairness metrics**: Does the model perform equally across demographic groups?

---

## 10. Interview Questions

**Q1 (Beginner): What is the difference between accuracy and F1 score?**
- Strong: "Accuracy is the fraction of all predictions that are correct. F1 is the harmonic mean of precision and recall. For balanced multi-class datasets like AG News, they're similar. But for imbalanced datasets, accuracy can be misleading — a model predicting the majority class gets high accuracy but fails on the minority class. F1 captures both false positives and false negatives."

**Q2 (Intermediate): This project uses macro-averaged F1. When would you use weighted-average F1 instead?**
- Strong: "Macro F1 treats each class equally. If class A has 100 samples and class B has 10,000, a bad macro F1 on class A pulls the overall score down equally. Weighted F1 weights each class by its support size — so class B has 100x more influence. Use weighted when larger classes are more business-important. Use macro when every class matters equally, regardless of size."

**Q3 (Advanced): When should a model be retrained? Give a precise decision process.**
- Strong: Walk through the 5-step framework above. Key points: (1) drift alone is not enough — you need performance impact, (2) you need a labeled sample of recent data to measure real performance, (3) proxy metrics (prediction distribution shift) can serve as leading indicators before labels arrive.

---

## Must Know
- Accuracy = correct / total (misleading for imbalanced classes)
- Precision = of what I predicted, how much was right
- Recall = of all actual positives, how many did I find
- F1 = harmonic mean of precision and recall
- AUC-ROC = discriminative ability across all thresholds (0.5=random, 1.0=perfect)
- Macro avg = equal weight per class, weighted avg = weighted by support
- Always use TEST set for final reporting, VALIDATION set for model selection

## Should Know
- Multi-class AUC-ROC uses One-vs-Rest strategy
- When to prefer PR-AUC over ROC-AUC (highly imbalanced)
- How to read a classification report (precision/recall/f1/support per class)
- Confusion matrix interpretation (off-diagonal = confusion between classes)
- The retraining decision framework

## Nice to Know
- Calibration (Brier score, calibration plots)
- Statistical significance testing (DeLong test for AUC comparison)
- Fairness metrics (equalized odds, demographic parity)
- Cost-sensitive learning
