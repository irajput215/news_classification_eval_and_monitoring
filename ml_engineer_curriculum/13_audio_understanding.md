# TOPIC 13: Audio Understanding

---

## 1. The Audio Pipeline

```
Real-world sound (pressure waves)
         ↓
Microphone (analog → digital signal)
         ↓
Digital waveform (sequence of amplitude samples)
         ↓
Preprocessing (normalization, resampling, VAD)
         ↓
Feature extraction (STFT → Mel spectrogram → MFCC)
         ↓
Audio encoder (CNN / Transformer)
         ↓
Audio representation (embeddings)
         ↓
Downstream task (ASR, classification, audio QA)
```

---

## 2. The Waveform

A digital waveform is a sequence of amplitude values over time.

```
x[n] = amplitude at sample n

Key parameters:
  Sample rate (fs): samples per second (Hz)
    16,000 Hz = 16 kHz → standard for speech
    22,050 Hz → standard for music (Whisper uses 16kHz)
    44,100 Hz → CD quality
  
  Bit depth: value range
    16-bit → values in [-32768, 32767]
    float32 → normalized to [-1.0, 1.0]
  
  Duration: n_samples / sample_rate
    16,000 samples at 16kHz = 1 second of audio
```

```python
import librosa
import numpy as np

# Load audio file (resamples to target sr automatically)
waveform, sr = librosa.load("audio.mp3", sr=16000)  # mono, 16kHz
print(f"Duration: {len(waveform) / sr:.2f} seconds")
print(f"Waveform shape: {waveform.shape}")   # (n_samples,)
print(f"Value range: [{waveform.min():.3f}, {waveform.max():.3f}]")
```

---

## 3. FFT and STFT

### FFT (Fast Fourier Transform)
Converts a time-domain signal into frequency domain — tells you WHICH frequencies are present.

```python
import numpy as np

# FFT of full waveform
fft = np.fft.rfft(waveform)   # real FFT (audio is real-valued)
magnitudes = np.abs(fft)       # frequency magnitudes
frequencies = np.fft.rfftfreq(len(waveform), d=1/sr)

# But FFT gives a global frequency view — we lose WHEN frequencies occur
```

**Problem with raw FFT:** Speech changes over time (phonemes, words). FFT loses temporal information.

### STFT (Short-Time Fourier Transform)
Apply FFT to overlapping windows:

```
Waveform: [----window_1----]
                   [----window_2----]
                           [----window_3----]
                                   ...
Each window → FFT → frequency snapshot
Stack all snapshots → 2D spectrogram
```

```python
import librosa

# STFT parameters:
n_fft = 1024          # FFT window size (determines frequency resolution)
hop_length = 256      # step between windows (determines time resolution)
# time resolution = hop_length / sr = 256 / 16000 = 16ms per frame

# Compute STFT
stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)   # magnitudes only
# shape: (1 + n_fft//2, n_frames) = (513, n_frames) for n_fft=1024

# Convert to dB (log scale — matches human hearing)
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
```

---

## 4. Mel Spectrogram

Human hearing is not linear in frequency — we're more sensitive to differences at low frequencies.

The **Mel scale** maps linear frequency to a perceptually uniform scale:
```
mel = 2595 × log10(1 + hz/700)
```

A **Mel filterbank** applies triangular filters spaced on the Mel scale:
```
Linear freq: 0, 250, 500, 750, 1000, 2000, 4000, 8000 Hz
Mel freq:    0, 50,  100, 150, 200,  282,  400,  550  mels
```

```python
# Mel spectrogram (what Whisper, Wav2Vec2, and most models use)
mel_spec = librosa.feature.melspectrogram(
    y=waveform,
    sr=sr,
    n_mels=80,            # 80 mel frequency bins (Whisper default)
    n_fft=400,            # Whisper uses 400 (25ms at 16kHz)
    hop_length=160,       # Whisper uses 160 (10ms stride)
    fmin=0,
    fmax=8000,
)

# Convert to log scale (log mel spectrogram)
log_mel = librosa.power_to_db(mel_spec)
print(f"Mel spectrogram shape: {log_mel.shape}")
# (80, n_frames) → (80, 3000) for 30 seconds of audio
```

**Why 80 mel bins?** This captures enough frequency detail for speech while keeping the feature dimension tractable. Whisper uses exactly 80.

---

## 5. MFCCs

MFCCs (Mel Frequency Cepstral Coefficients) apply a DCT to the log-mel spectrogram:

```python
mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
# shape: (13, n_frames)
```

**Why MFCC?** Compresses spectral information, decorrelates features, very compact.

**Why NOT MFCC for modern DL?** Deep learning models learn better representations directly from mel spectrograms. MFCC is mostly used in classical speech processing (HMM-based ASR, GMMs).

**Modern preference:** Use mel spectrograms directly (Whisper, Wav2Vec2). Skip MFCC.

---

## 6. Voice Activity Detection (VAD)

Detect speech vs silence:

```python
# Using Silero VAD (fast, accurate)
import torch

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad"
)
(get_speech_timestamps, _, read_audio, *_) = utils

wav = read_audio("audio.wav", sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# → [{'start': 0, 'end': 8000}, {'start': 12000, 'end': 20000}, ...]
# Each entry: start/end sample indices of speech segments
```

**Why VAD matters:**
- Remove silence before sending to ASR → 30-50% faster transcription
- Segment long audio into chunks (ASR models have max context limits)
- Required for real-time streaming ASR

---

## 7. Audio Normalization and Chunking

```python
def preprocess_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Standard preprocessing pipeline for ASR."""
    # 1. Load and convert to mono
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # 2. Normalize amplitude to [-1, 1]
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp
    
    return waveform

def chunk_audio(waveform: np.ndarray, sr: int,
                chunk_duration: float = 30.0,
                overlap_duration: float = 2.0) -> list:
    """Split long audio into overlapping chunks for ASR."""
    chunk_size = int(chunk_duration * sr)    # e.g., 30 * 16000 = 480000 samples
    overlap_size = int(overlap_duration * sr)  # 2 * 16000 = 32000 samples
    
    chunks = []
    start = 0
    while start < len(waveform):
        end = min(start + chunk_size, len(waveform))
        chunks.append({
            "audio": waveform[start:end],
            "start_time": start / sr,
            "end_time": end / sr,
        })
        start += chunk_size - overlap_size  # overlap to avoid cutting words
    return chunks
```

**Why overlap?** Words at chunk boundaries get cut. With overlap, each word appears in at least one complete chunk. During transcription, the overlap region is merged/deduplicated.

---

## 8. Interview Questions

**Q1 (Beginner): What is a Mel spectrogram and why do ASR models use it instead of raw waveforms?**
Strong: "A mel spectrogram represents audio as a 2D image: frequency (mel scale, perceptually uniform) on the Y axis, time on the X axis, and intensity as pixel value. Raw waveforms have 16,000 samples per second — that's a very high-dimensional sequence that's hard to learn from efficiently. A mel spectrogram compresses this into ~100 frames per second × 80 mel bins = 8,000 values/second — a much more compact and perceptually meaningful representation. The mel scale also matches human hearing, so features that matter acoustically get more resolution."

**Q2 (Intermediate): Why is VAD important before ASR, and what happens without it?**
Strong: "Without VAD, silence regions are sent to the ASR model, which may generate hallucinations — the model 'hears' non-existent words in silence. More practically, silence pads the context window, reducing effective throughput. With VAD, you extract only speech segments, send those to ASR, and map timestamps back to original audio. This reduces ASR compute by 30-50% on typical conversational audio (which is 40-50% silence) and eliminates silence hallucinations."

---

## MUST KNOW Summary
- Sample rate: 16kHz standard for speech
- STFT: sliding FFT over windows → time × frequency 2D representation
- Mel spectrogram: STFT with perceptually-uniform frequency binning (80 bins standard)
- Log scale: always apply log to mel spectrogram (models train better on log-scale)
- VAD: detect and remove silence before ASR → faster, no hallucinations
- Chunking: split audio > 30s into overlapping chunks
- MFCC: deprecated for DL; use mel spectrogram directly
