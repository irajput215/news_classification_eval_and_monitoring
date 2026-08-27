# TOPIC 15: Forced Alignment

---

## 1. What Forced Alignment Is

```
ASR (no transcript given):
  Audio → Model → Transcript
  "What did they say?"

Forced Alignment (transcript given):
  Audio + Transcript → Model → Timestamps
  "Where exactly does each word start and end?"
```

You KNOW what was said. You want to know WHEN each word (or phoneme) occurs.

**Applications:**
- Subtitle synchronization (find where "hello" occurs in a 10-minute video)
- Audio dataset creation (pair audio segments with text labels)
- Voice acting evaluation (did actor say the line at the right time?)
- Karaoke / lyrics sync
- Training data for TTS models

---

## 2. The CTC Trellis

CTC-based forced alignment exploits the per-frame character probabilities from a CTC ASR model.

### Step 1: Run CTC encoder
```
Audio frames: [f1, f2, f3, ..., fT]
                    ↓
CTC model (Wav2Vec2 or similar) produces:
  p[t][c] = probability of character c at frame t
  Shape: (T frames, vocab_size)

Example for audio saying "cat":
  Frame 1: P("c")=0.9, P("-")=0.08, ...
  Frame 2: P("a")=0.7, P("c")=0.2, ...
  Frame 3: P("a")=0.8, P("-")=0.15, ...
  Frame 4: P("t")=0.85, P("-")=0.1, ...
  ...
```

### Step 2: Build the trellis
A trellis is a 2D matrix where:
- Rows = positions in the target transcript (including blanks)
- Columns = time frames

For target "cat" with blanks: `- c - a - t -`

```
         frame1  frame2  frame3  frame4  frame5
"-"     [ 0.08 ]
"c"              [ 0.90 ]
"-"                      [ 0.08 ]
"a"                               [ 0.70 ]
"-"                                        [ 0.15 ]
"t"              ...
"-"              ...
```

### Step 3: Dynamic programming (Viterbi-like)
Find the most probable path through the trellis that:
1. Starts at the first blank or first character
2. Ends at the last character or last blank
3. Only moves forward in the transcript (no skipping)
4. Can stay in the same position (repeated character = same position)

```python
import torch
import numpy as np

def compute_trellis(emission: torch.Tensor, tokens: list, blank_id: int) -> torch.Tensor:
    """
    emission: (T, vocab_size) — CTC log-probabilities
    tokens: list of token ids for the target text (with blanks inserted)
    Returns: trellis (T, len(tokens)) of cumulative log-probabilities
    """
    T, vocab_size = emission.shape
    num_tokens = len(tokens)
    
    # Initialize
    trellis = torch.full((T + 1, num_tokens + 1), -float("inf"))
    trellis[0, 0] = 0
    
    for t in range(T):
        trellis[t + 1, 1:] = torch.maximum(
            # Stay in same position
            trellis[t, 1:] + emission[t, tokens],
            # Move to next position (if valid)
            trellis[t, :-1] + emission[t, tokens]
        )
    return trellis
```

### Step 4: Backtrack to find alignment
Trace the path with maximum probability backwards from the end.

---

## 3. Wav2Vec2-Based Alignment (torchaudio MFA)

The cleanest approach in modern Python:

```python
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle

# Load Wav2Vec2 alignment model
model = bundle.get_model()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

# Load audio
waveform, sr = torchaudio.load("audio.wav")
if sr != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

transcript = "hello my name is alice"

with torch.inference_mode():
    emission, _ = model(waveform)

tokenized_transcript = tokenizer(transcript)
word_spans = aligner(emission, tokenized_transcript)

# Get word-level timestamps
for word, span in zip(transcript.split(), word_spans):
    ratio = waveform.size(1) / emission.size(1)
    start_sample = int(span[0].start * ratio)
    end_sample = int(span[-1].end * ratio)
    start_time = start_sample / bundle.sample_rate
    end_time = end_sample / bundle.sample_rate
    print(f"{word}: {start_time:.3f}s → {end_time:.3f}s")
```

Output:
```
hello: 0.040s → 0.320s
my: 0.340s → 0.440s
name: 0.460s → 0.620s
is: 0.640s → 0.720s
alice: 0.740s → 1.040s
```

---

## 4. WhisperX (Whisper + Alignment)

WhisperX combines Whisper's excellent ASR with a separate forced alignment step:

```
Step 1: Whisper transcribes audio → word-level timestamps (rough, Whisper-native)
Step 2: Wav2Vec2 alignment model refines timestamps (precise, phoneme-level)
```

```bash
pip install whisperx

# Command line
whisperx audio.mp3 --model large-v3 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H

# Python
import whisperx

model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")
result = model.transcribe("audio.mp3", batch_size=16)

# Load alignment model
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, "audio.mp3", device="cuda")

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.3f}s → {word['end']:.3f}s")
```

WhisperX is the practical production choice for word-level timestamps.

---

## 5. Montreal Forced Aligner (MFA)

MFA is a professional-grade aligner used in linguistics and speech research:
- Works at PHONEME level (not just word level)
- Uses pre-trained acoustic models + pronunciation dictionaries
- Handles speaker-specific acoustic variation

```bash
# Install MFA
pip install montreal-forced-aligner

# Download English acoustic model + dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Align a directory of audio + transcript files
# (each audio file must have a corresponding .txt file with the transcript)
mfa align \
    /path/to/audio_directory \     # contains: utterance1.wav, utterance1.txt
    english_us_arpa \              # pronunciation dictionary
    english_us_arpa \              # acoustic model
    /path/to/output               # TextGrid files with alignment
```

Output: TextGrid files (Praat format) containing phoneme and word intervals.

---

## 6. Pronunciation Dictionaries

For phoneme-level alignment, you need to know HOW words are pronounced.

```
"hello" → HH AH0 L OW1
"world" → W ER1 L D
```

CMU Pronouncing Dictionary is the standard for English:
```python
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

d = cmudict.dict()
print(d["hello"])
# → [['HH', 'AH0', 'L', 'OW1']]

# OOP words with multiple pronunciations
print(d["either"])
# → [['IY1', 'DH', 'ER0'], ['AY1', 'DH', 'ER0']]
```

---

## 7. ASR vs Forced Alignment — Summary Table

| | ASR | Forced Alignment |
|--|-----|-----------------|
| Input | Audio only | Audio + known transcript |
| Output | Transcript | Timestamps for each word/phoneme |
| When you know what was said | No (infer) | Yes (given) |
| Accuracy | ~5% WER for clear speech | Millisecond precision |
| Use case | Transcription, captioning | Subtitle sync, dataset creation, evaluation |
| Models | Whisper, Wav2Vec2 | WhisperX, MFA, torchaudio aligner |

---

## 8. Interview Questions

**Q1 (Intermediate): What is the difference between ASR and forced alignment?**
Strong: "ASR answers 'what was said?' — it takes audio and produces a transcript. Forced alignment answers 'when was each word said?' — given audio AND a known transcript, it finds the precise timestamp for each word (and optionally each phoneme). Forced alignment is more accurate for timing because the search space is constrained by the known transcript, and models like MFA work at phoneme level with pronunciation dictionaries. The CTC trellis algorithm finds the most probable alignment via dynamic programming."

**Q2 (Advanced): Walk me through how CTC produces word timestamps during forced alignment.**
Strong: "The CTC encoder (e.g., Wav2Vec2) produces per-frame log-probabilities over the character vocabulary — for each of the T audio frames, you get a distribution over all characters. For a known transcript like 'cat', you construct a sequence including blanks: [blank, c, blank, a, blank, t, blank]. Then you build a trellis: a T × len(tokens) matrix. Dynamic programming fills this trellis, accumulating log-probabilities along all valid paths (same position or advance one position). Backtracking from the end gives the most probable path. Where a character's path segment starts and ends in time gives its timestamp."

---

## MUST KNOW Summary
- Forced alignment = audio + known transcript → word timestamps
- CTC trellis = 2D DP matrix over (time, transcript position) pairs
- WhisperX = Whisper ASR + Wav2Vec2 alignment (practical production choice)
- MFA = phoneme-level alignment, linguistics standard, needs pronunciation dict
- Applications: subtitle sync, training data creation, performance evaluation
