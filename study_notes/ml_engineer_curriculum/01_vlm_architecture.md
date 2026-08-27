# TOPIC 01: Vision-Language Model Architecture

---

## WHY

You cannot fine-tune what you don't understand. Before touching LoRA or DPO, you must know:
- Where does an image actually go in the model?
- How does the LLM "see" the image?
- What is trainable and what is frozen?
- Why do different VLMs make different architectural trade-offs?

Without this, you will blindly copy config files and not know why training fails.

---

## 1. What is a Vision-Language Model?

A VLM is a model that takes **images + text** as input and produces **text** as output.

NOT a diffusion model (does not generate images).
NOT a CLIP model (does not produce embeddings only).

A VLM is fundamentally: **a large language model that has learned to understand images as tokens.**

The key insight: **LLMs are sequence models. Images must become a sequence of tokens.**

---

## 2. The Full Architecture

```
Input:
  [Image: 448×448 RGB] + [Text: "Describe this image in detail."]
        |                          |
        ↓                          ↓
  ┌─────────────────┐        ┌─────────────┐
  │  Vision Encoder │        │  Tokenizer  │
  │  (ViT / SigLIP) │        │             │
  └────────┬────────┘        └──────┬──────┘
           │                        │
           │ Visual patch           │ Text tokens
           │ embeddings             │ [token_ids]
           │ shape: (n_patches, d)  │
           ↓                        │
  ┌─────────────────┐               │
  │   Projector /   │               │
  │   Connector     │               │
  │ (MLP or cross-  │               │
  │  attention)     │               │
  └────────┬────────┘               │
           │                        │
           │ Visual tokens          │
           │ shape: (n_visual, d_llm)│
           ↓                        ↓
  ┌──────────────────────────────────────┐
  │          Large Language Model        │
  │   [img_tok_1, ..., img_tok_n,        │
  │    text_tok_1, ..., text_tok_m]      │
  │                                      │
  │   Self-attention over all tokens     │
  └───────────────────────┬──────────────┘
                          │
                          ↓
                   [Text output tokens]
                   "A photo showing..."
```

---

## 3. Vision Encoder

### What it is
The vision encoder converts a raw image into a sequence of dense vector representations (embeddings). Almost all modern VLMs use a **Vision Transformer (ViT)** as the encoder.

### How ViT works

```
Image: 448 × 448 × 3
         ↓
Divide into patches (e.g., 14×14 patches each of size 32×32)
         ↓
Flatten each patch → vector of 32×32×3 = 3072 values
         ↓
Linear projection → 1024-dim embedding
         ↓
Add position embeddings (so model knows where patch is)
         ↓
Feed through N transformer layers
         ↓
Output: sequence of patch embeddings
        shape: (n_patches, d_model)
        e.g., (1024, 1024) for 32×32 patches on 1024×1024 image
```

**Patch count formula:**
```
n_patches = (image_height / patch_size) × (image_width / patch_size)
          = (448 / 14) × (448 / 14)
          = 32 × 32
          = 1024 patches
```

Each patch becomes ONE embedding in the sequence → 1024 visual "tokens" before the projector.

### Real encoders used

| Encoder | Used in | Patch size | Notes |
|---------|---------|------------|-------|
| ViT-L/14 | LLaVA-1.5 | 14×14 | OpenAI CLIP visual encoder |
| SigLIP-400M | PaliGemma, Gemma 3 multimodal | 14×14 | Google's CLIP variant, better for vision |
| ViT-H/14 | Qwen2-VL | 14×14 | Large 600M encoder |
| SigLIP-SO400M | SmolVLM | 14×14 | Same family, smaller LLM |

### Key insight: CLIP-trained vs SigLIP-trained encoders

**CLIP encoders** were trained with contrastive loss on (image, text) pairs — they learned to make matching pairs similar and non-matching pairs different. Good at image-text alignment. Less good at fine-grained spatial understanding.

**SigLIP** uses a sigmoid binary cross-entropy loss instead of softmax, which scales better and produces better vision representations.

---

## 4. Projector / Connector

### The problem
- Vision encoder outputs dimension: `d_vision` (e.g., 1024)
- LLM expects dimension: `d_llm` (e.g., 4096 for a 7B model)
- These don't match

The projector bridges them.

### Types of projectors

**MLP Projector (LLaVA-1.5 style)**
```python
# Two-layer MLP with GELU
projector = nn.Sequential(
    nn.Linear(d_vision, d_llm),
    nn.GELU(),
    nn.Linear(d_llm, d_llm)
)
```
Simple, fast, effective. Each visual token is independently projected.

**Pixel Shuffle / Vision Resampler (Qwen2-VL style)**
```
1024 visual patches (before)
    ↓
Pixel Shuffle (2×2) — merges 4 patches into 1
    ↓
256 visual tokens (after)
    ↓
MLP projection
```
Reduces visual token count → shorter context → faster training and inference.

**Cross-attention (Flamingo / InstructBLIP style)**
```
LLM uses cross-attention to "query" visual features
Visual features stay separate from LLM token stream
→ Fewer visual tokens inserted into context
→ More flexible for many images
```

### Why projector design matters

More visual tokens → richer image understanding but slower training and inference.
Fewer visual tokens → faster but may lose spatial detail.

Qwen2-VL's pixel shuffle is a key innovation — it balances resolution and speed.

---

## 5. Language Model Backbone

The LLM receives a concatenated sequence:
```
[visual_token_1, visual_token_2, ..., visual_token_N, text_token_1, ..., text_token_M]
```

The LLM is completely standard — it's the same transformer that processes text-only inputs. The ONLY change is that some of the input tokens come from the vision pathway instead of the text embedding table.

Standard decoder-only LLMs used:
- Qwen2-VL: Qwen2 (72B, 7B, 2B)
- LLaVA-1.5: Vicuna / Mistral / LLaMA
- PaliGemma: Gemma 2B
- SmolVLM: SmolLM (135M–1.7B)
- Gemma 3 multimodal: Gemma 3 (4B, 12B, 27B)

---

## 6. Multimodal Tokenization

### Text tokenization (review)
Text → tokenizer → list of integer token IDs → embedding lookup table → embeddings

### Image tokenization
Image → vision encoder → patch embeddings → projector → visual embeddings

These visual embeddings are inserted into the sequence at the position of a **special image token** in the text.

### Qwen2-VL example
```
User text: "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What's in this image?<|im_end|>"

During processing:
- <|image_pad|> is a placeholder
- The N visual embeddings replace this placeholder
- The final sequence seen by the LLM has N image tokens + text tokens
```

### LLaVA example
```
User: "USER: <image>\nDescribe this image. ASSISTANT:"
- <image> is a single special token
- During forward pass, N visual tokens replace this single token
```

### What happens in the model's self-attention
ALL tokens — visual and text — attend to each other bidirectionally within the visual context, and the text attends causally after the image. This is why the LLM "sees" the image: the attention mechanism allows text tokens to look at visual patch tokens.

---

## 7. Image Preprocessing

Before the vision encoder, images must be standardized.

### Standard pipeline
```python
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((448, 448)),          # standardize size
    transforms.ToTensor(),                  # [H, W, C] → [C, H, W], values 0-1
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet mean (for CLIP-based encoders)
        std=[0.229, 0.224, 0.225]           # ImageNet std
    )
])
```

### Why normalization?
Vision encoders are pre-trained on images normalized to a specific mean/std. If you change these values, the encoder's pre-trained weights produce garbage outputs. Always use the same normalization used during encoder pre-training.

### Dynamic resolution (Qwen2-VL innovation)
Instead of resizing ALL images to 448×448, Qwen2-VL supports variable resolutions:
- Small image: fewer patches → shorter sequence
- Large/detailed image: more patches → longer sequence (up to 1280 patches)

This preserves native image resolution, which is critical for tasks like OCR and fine-grained visual understanding.

### AnyRes / LLaVA-HD
LLaVA-1.6 introduced "AnyRes" — process an image at multiple resolutions:
1. Base view: resized to 336×336 (standard)
2. Additional crops: divide image into tiles and process each tile

This gives more visual tokens for detailed images without retraining the base model.

---

## 8. Instruction Tuning for VLMs

**Instruction tuning** = supervised fine-tuning on (instruction, response) pairs.

For VLMs, the instruction can contain images.

### Data format (Qwen2-VL style)
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The image shows a street scene..."}
            ]
        }
    ]
}
```

### What the loss function trains on
```
Full sequence: [system] [user: image + question] [assistant: answer]
                                                   ↑
                                         Loss computed ONLY here
```

Loss is computed ONLY on the assistant's response tokens. The model is not penalized for what it generates while "reading" the image and question.

### Why this matters for VLMs specifically
During SFT, you must be careful about which tokens compute loss:
- Visual tokens: NO loss (these are inputs, not predictions)
- User text tokens: NO loss (same reason)
- Assistant tokens: YES loss (these are what the model is learning to generate)

In HuggingFace TRL's `SFTTrainer`, this is controlled by `response_template` or the `DataCollatorForCompletionOnlyLM`.

---

## 9. Trainable Parameters by Scenario

This is one of the most important tables for interviews:

| Scenario | Frozen | Trainable | When to use |
|----------|--------|-----------|-------------|
| Zero-shot | Everything | Nothing | Just using the model |
| Adapter only (LoRA on LLM) | Vision encoder + Projector + LLM weights | LoRA matrices | Most common SFT scenario |
| Full LoRA | Vision encoder | Projector + LoRA on LLM | When projector needs alignment to new domain |
| Projector-only tuning | Vision encoder + LLM | Projector | Stage 1 of LLaVA training |
| Full fine-tuning | Nothing | Everything | Large dataset, high resources |
| LoRA on all modules | Nothing frozen | LoRA everywhere | Aggressive but memory-heavy |

### The LLaVA two-stage training approach
```
Stage 1: Train ONLY the projector
- Dataset: 595K image-caption pairs
- Goal: align visual embeddings to LLM embedding space
- Vision encoder: FROZEN
- LLM: FROZEN
- Projector: TRAINABLE

Stage 2: Instruction tuning with LoRA on LLM
- Dataset: 150K instruction-following pairs
- Goal: teach the model to follow visual instructions
- Vision encoder: FROZEN
- Projector: FROZEN (or lightly tuned)
- LLM: LoRA adapters trained
```

---

## 10. Concrete Models — What to Know

### Qwen2-VL (Recommended for interview focus)
```
Architecture: ViT (ViT-BigG) + MLP projector + Qwen2 LLM
Sizes: 2B, 7B, 72B
Key innovation: Naive Dynamic Resolution (NaViT) — variable patch count
Context: 32K tokens
Multimodal: image + video + text
HuggingFace: Qwen/Qwen2-VL-7B-Instruct
```

### LLaVA-1.5 (Good learning model — cleaner architecture)
```
Architecture: CLIP ViT-L/14@336 + 2-layer MLP projector + Vicuna/Mistral
Sizes: 7B, 13B
Key innovation: MLP connector (vs simpler linear in v1)
Training data: ShareGPT4V, LLaVA-Instruct-150K
HuggingFace: llava-hf/llava-1.5-7b-hf
```

### SmolVLM (Best for rapid experimentation)
```
Architecture: SigLIP-400M + MLP + SmolLM 256M/500M/1.7B
Size: 256M, 500M, 2B total
Key advantage: fits on CPU, fast iteration, good for learning
HuggingFace: HuggingFaceTB/SmolVLM-Instruct
```

### Gemma 3 Multimodal
```
Architecture: SigLIP + Gemma 3 LLM
Sizes: 4B, 12B, 27B
Key: Strong vision understanding, good instruction following
HuggingFace: google/gemma-3-4b-it
```

---

## 11. How to Load and Inspect a VLM

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# Load model + processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Inspect architecture
print(model)
# → Qwen2VLForConditionalGeneration(
#     visual: Qwen2VisionTransformerPretrainedModel(...)  ← vision encoder
#     model: Qwen2VLModel(                               ← LLM
#       embed_tokens: Embedding(...)
#       layers: 28 × Qwen2VLDecoderLayer(...)
#     )
# )

# Count parameters
total = sum(p.numel() for p in model.parameters()) / 1e9
vision = sum(p.numel() for p in model.visual.parameters()) / 1e9
llm = total - vision
print(f"Total: {total:.1f}B | Vision: {vision:.1f}B | LLM: {llm:.1f}B")
```

---

## 12. Failure Modes

### Training fails / loss doesn't decrease
- Wrong loss mask (computing loss on image tokens or user tokens)
- Image normalization mismatch (different mean/std than pre-training)
- Learning rate too high (blows up vision-language alignment)

### Model generates garbage after fine-tuning
- Training-serving preprocessing mismatch (different image size or normalization)
- Chat template not applied correctly at inference
- Forgetting to include the `<image>` or equivalent token

### OOM during training
- Too many visual tokens per sample (high resolution images)
- Batch size too large
- Not using gradient checkpointing

### Model forgets language capabilities (catastrophic forgetting)
- Learning rate too high
- Not enough diversity in SFT dataset
- Fine-tuning ALL parameters instead of LoRA

---

## 13. Repository Deep Dive

### Primary: LLaVA
```
Repo: https://github.com/haotian-liu/LLaVA
Key files:
  llava/model/llava_arch.py    ← how visual tokens are inserted into LLM
  llava/model/multimodal_encoder/clip_encoder.py ← vision encoder wrapper
  llava/model/multimodal_projector/builder.py ← projector architectures
  llava/train/train.py         ← training loop, loss masking

Experiment: Find where visual token loss masking happens.
Search for: IGNORE_INDEX
Expected: image tokens set to IGNORE_INDEX → not included in loss
```

### Primary: Qwen2-VL
```
Repo: https://github.com/QwenLM/Qwen2-VL
Key files:
  qwen_vl_utils/vision_process.py  ← image preprocessing, dynamic resolution
  modeling_qwen2_vl.py (in HF)    ← full model definition

Experiment: Load Qwen2-VL-2B-Instruct, process one image, print the
  shape of visual embeddings before and after the projector.
Expected: (n_patches, 1280) → (n_visual_tokens, 4096)
```

### Also study:
```
HuggingFace TRL VLM SFT example:
https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py
```

---

## 14. Interview Questions

**Q1 (Beginner): What is the role of the projector in a VLM?**
Strong: "The projector bridges the dimension gap between the vision encoder's output and the LLM's expected input. The vision encoder might output 1024-dimensional embeddings, but a 7B LLM expects 4096-dimensional token embeddings. The projector — typically a 2-layer MLP — linearly projects visual patch embeddings to the LLM's dimension, enabling the LLM to process them as if they were regular tokens."

**Q2 (Intermediate): A VLM's text generation quality dropped after you fine-tuned it on a custom image dataset. What could cause this?**
Strong: "Catastrophic forgetting — if we fine-tuned the LLM weights fully, the model overwrites its text understanding. Fix: use LoRA instead of full fine-tuning, which preserves base weights. Second possibility: the SFT dataset is text-poor — model drifts toward image-heavy responses. Third: wrong loss masking — if loss is computed on image tokens or user instructions, the model receives incorrect gradient signal."

**Q3 (Advanced): Why does Qwen2-VL use dynamic resolution instead of resizing all images to a fixed size?**
Strong: "Fixed resizing to 448×448 loses information for high-resolution images (like an invoice or a dense chart) while being wasteful for small images. Dynamic resolution processes each image at its native resolution by generating variable numbers of patches. A 1024×1024 image gets 4× more patches than a 224×224 image. This costs more tokens but preserves fine-grained details needed for OCR, chart reading, and spatial reasoning. The model learns to handle variable-length visual sequences through positional embeddings that encode 2D position, not just 1D sequence position."

---

## MUST KNOW Summary
- VLM = Vision Encoder + Projector + LLM
- Images become sequences of visual tokens via ViT patches
- Projector maps vision dimensions to LLM dimensions
- Loss is computed ONLY on assistant response tokens
- Visual tokens in context but not in loss
- LoRA on LLM weights is the standard fine-tuning approach
- Dynamic resolution (Qwen2-VL) vs fixed resolution (LLaVA-1.5) trade-off
