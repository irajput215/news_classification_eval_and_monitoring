# TOPIC 14: Automatic Speech Recognition (ASR)

---

## 1. The ASR Pipeline

```
Audio waveform
     ↓
Preprocessing (16kHz, mono, normalized)
     ↓
Feature extraction (80-dim log-mel spectrogram)
     ↓
Audio Encoder (CNN + Transformer or pure Transformer)
     ↓
Acoustic representations (contextual embeddings of audio)
     ↓
Decoder mechanism (CTC / Attention decoder / RNN-T)
     ↓
Transcript (sequence of characters/subwords/words)
```

---

## 2. CTC (Connectionist Temporal Classification)

### What it solves
Transcribing speech requires ALIGNMENT — which audio frame corresponds to which character? But we don't have frame-level alignment labels, only the full transcript.

CTC solves this without requiring alignment labels.

### How CTC works

```
Audio frames: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
                ↓
Each frame gets a probability distribution over vocabulary + blank token
  f1: "c"=0.8, "-"=0.1, other=0.1
  f2: "a"=0.7, "-"=0.2, other=0.1
  f3: "a"=0.6, "-"=0.3, other=0.1    ← same character
  f4: "t"=0.9, "-"=0.05, other=0.05
  ...

CTC decoding collapses:
  "c", "a", "a", "t", "-", "-", "s" → "cats"
  (merge repeated characters, remove blanks)
```

**CTC blank token "-":** The model can emit blank when nothing new should be output. This allows the model to handle the timing mismatch between slow audio frames and fast character output.

### CTC Loss
Sums probability over ALL valid alignments between the output sequence and target:
```
L_CTC = -log P(target | audio) = -log Σ_{valid alignments} P(alignment | audio)
```

Computed efficiently via dynamic programming on the lattice of all alignments.

### CTC for alignment
CTC is not just for ASR — it's used for forced alignment (Topic 15) because it produces per-frame character probabilities that encode timing.

---

## 3. Encoder-Decoder ASR (Whisper)

### Architecture
```
Audio → Log-mel spectrogram → Encoder (Transformer) → Encoded representations
                                                              ↓
                                       ← Text tokens ← Decoder (Transformer)
                                         (autoregressive)
```

Whisper uses standard encoder-decoder transformer architecture (like the original Attention is All You Need paper).

### Whisper specifics
```
Input: 30-second audio chunks (zero-pad shorter)
Input features: 80-dim log-mel spectrogram, 3000 frames (10ms stride)
Encoder: 6-32 transformer layers (depends on model size)
Decoder: 6-32 transformer layers
Vocabulary: 51,865 tokens (multilingual BPE)

Model sizes:
  tiny:   39M params, ~32× realtime
  base:   74M params, ~16× realtime
  small:  244M params, ~6× realtime  ← good trade-off
  medium: 769M params, ~2× realtime
  large-v3: 1.55B params, ~1× realtime
```

### Using Whisper
```python
import whisper

# Load model
model = whisper.load_model("large-v3")

# Basic transcription
result = model.transcribe("audio.mp3")
print(result["text"])   # full transcript

# With language detection
result = model.transcribe("audio.mp3", task="transcribe", language="en")
print(result["language"])   # "en"

# With word timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

### Faster Whisper (for production)
```python
from faster_whisper import WhisperModel

# 4-bit quantized, 2× faster
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
```

---

## 4. Wav2Vec 2.0

Wav2Vec 2.0 is different from Whisper — it's a self-supervised model that learns speech representations.

```
Waveform
    ↓
Feature encoder (CNN): local feature extraction from raw audio
    ↓
Context network (Transformer): long-range context
    ↓
Pre-training: contrastive loss (predict quantized representations)
    ↓
Fine-tuning: CTC head added, trained on (audio, transcript) pairs
```

### Key difference from Whisper:
- Wav2Vec 2.0: encoder-only, uses CTC head → faster inference, no beam search needed
- Whisper: encoder-decoder, autoregressive → slower but better accuracy on difficult audio

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Preprocess
waveform, sr = librosa.load("audio.wav", sr=16000)
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

# Inference
with torch.no_grad():
    logits = model(**inputs).logits   # shape: (batch, time, vocab)

# Decode via CTC greedy
predicted_ids = torch.argmax(logits, dim=-1)
transcript = processor.decode(predicted_ids[0])
print(transcript)
```

---

## 5. Transducer (RNN-T)

Transducers combine acoustic and language model in a unified framework. Used in:
- Google's speech recognition (Conformer-T)
- Apple's Siri
- Amazon Alexa

The key advantage over CTC: the transducer has an internal language model (predictor network) that conditions each output on all previous outputs. This gives much better accuracy.

The key advantage over encoder-decoder: naturally handles streaming audio (no need to wait for end of utterance before decoding starts).

---

## 6. WER (Word Error Rate)

The primary metric for ASR evaluation.

### Formula
```
WER = (S + D + I) / N

S = Substitutions  (wrong word predicted where correct word exists)
D = Deletions      (correct word not predicted)
I = Insertions     (extra word predicted where none should be)
N = Number of words in reference transcript
```

### Calculation via dynamic programming
WER is essentially edit distance at the word level:
```python
from jiwer import wer

reference = "the cat sat on the mat"
hypothesis = "the cat sat on the"      # missing "mat"

# Word-level: S=0, D=1, I=0, N=6
# WER = 1/6 = 0.167 = 16.7%
print(wer(reference, hypothesis))   # 0.1667
```

### Worked example:
```
Reference:  "the quick brown fox jumps"
Hypothesis: "the quick brown dog jumps" (one substitution: fox→dog)

Align:
  REF:  the   quick  brown  fox   jumps
  HYP:  the   quick  brown  dog   jumps
  EDIT:  ✓     ✓      ✓      S     ✓

S=1, D=0, I=0, N=5
WER = 1/5 = 0.20 = 20%
```

### Another worked example:
```
Reference:  "hello world"
Hypothesis: "hello beautiful world today"  (insertion + insertion)

S=0, D=0, I=2, N=2
WER = 2/2 = 1.0 = 100%
(WER > 1.0 is possible! This means more errors than reference words)
```

### CER (Character Error Rate)
Same as WER but at the character level:
```python
from jiwer import cer
print(cer(reference, hypothesis))
```

**Use CER when:** Dealing with non-segmented languages (Chinese, Japanese), morphologically rich languages, or when measuring character-level accuracy for OCR tasks.

---

## 7. ASR in Production

```python
# Production ASR pipeline with VAD + chunking + Whisper

from faster_whisper import WhisperModel
import silero_vad  # simplified import

class ProductionASR:
    def __init__(self, model_size="large-v3"):
        self.whisper = WhisperModel(model_size, device="cuda", compute_type="int8")
        self.vad_model, self.vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad"
        )
    
    def transcribe(self, audio_path: str) -> list:
        # 1. Load and normalize
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # 2. VAD — get speech segments
        speech_timestamps = self._get_speech_timestamps(waveform, sr)
        
        # 3. Transcribe each speech segment
        results = []
        for seg in speech_timestamps:
            audio_chunk = waveform[seg["start"]:seg["end"]]
            
            segments, _ = self.whisper.transcribe(
                audio_chunk, beam_size=5, word_timestamps=True
            )
            
            # Adjust timestamps to global audio timeline
            offset = seg["start"] / sr
            for s in segments:
                results.append({
                    "text": s.text,
                    "start": s.start + offset,
                    "end": s.end + offset,
                })
        return results
```

---

## 8. Why WER Is Insufficient for Creative Audio

For standard news ASR: WER is fine — there's one correct transcript.

For evaluating creative speech (voice acting, audiobooks, dialogue performance):
- WER only measures word accuracy, not DELIVERY quality
- A perfectly accurate transcript can have wrong emotion, wrong rhythm, wrong emphasis
- WER cannot tell you: "The line was delivered too robotically"

**What to use instead:**
- Prosody analysis: pitch contour, speech rate, pause distribution
- Emotion classification: arousal + valence from audio
- Human evaluation: trained listener rates on rubric
- Speaker similarity: embedding distance from reference performance

---

## 9. Repositories

```
OpenAI Whisper:
  https://github.com/openai/whisper
  Key file: whisper/transcribe.py
  Experiment: transcribe a 1-min clip with --word_timestamps True, inspect alignment

Faster-Whisper (production inference):
  https://github.com/SYSTRAN/faster-whisper
  Key: 4-bit quantized CTranslate2 backend, 2-4× faster than openai/whisper

Wav2Vec 2.0:
  HuggingFace: facebook/wav2vec2-large-960h-lv60-self
  Experiment: Compare WER of Wav2Vec2 vs Whisper on LibriSpeech test-clean
```

---

## MUST KNOW Summary
- WER = (S+D+I)/N; can exceed 1.0; lower is better
- CTC: learns alignment without alignment labels; blank token handles timing
- Whisper: encoder-decoder, 30s chunks, 80-dim log-mel input, multiple sizes
- Wav2Vec2: encoder-only + CTC head, faster inference
- VAD before ASR: removes silence, prevents hallucinations, faster
- WER is insufficient for creative/expressive speech — need prosody + human eval
