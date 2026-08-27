# TOPIC 10: Checkpointing & Fault Tolerance

---

## WHY

A 24-hour training run on a spot instance will get interrupted. Without proper checkpointing, you lose everything. With checkpointing, you resume from 2 hours ago. The difference is losing $200 vs losing $4.

---

## 1. What a Checkpoint Contains

Saving only model weights is INSUFFICIENT for exact training recovery.

A complete checkpoint requires:

```
checkpoint/
├── model weights         ← the neural network parameters (after LoRA merge or just adapters)
├── optimizer state       ← AdamW's momentum (m) and variance (v) for every parameter
├── LR scheduler state    ← current lr, warmup progress, decay position
├── RNG state             ← random seed state for reproducibility
│   ├── Python random
│   ├── NumPy random
│   └── CUDA random (per GPU)
├── training step         ← which step we're on (not epoch alone)
├── epoch                 ← which epoch we're on
└── best metric           ← for best_model_at_end logic
```

**Why each component matters:**

| Component | What breaks without it |
|-----------|----------------------|
| Model weights | You don't have the trained model |
| Optimizer state | Adam restart means m=0, v=0. First many steps after resume have wrong gradient magnitudes — training "forgets" recent gradient history |
| LR scheduler | LR jumps back to initial value. You'd be at 1e-4 when you should be at 2e-5 (cosine decay) |
| RNG state | Training is no longer reproducible; with data shuffling, next epoch will use different order |
| Training step | Can't correctly continue epoch or warmup |

---

## 2. The Fault Tolerance Pattern

```
Training start
    ↓
Step 1: data + forward + backward + optimizer.step()
    ↓
Step 2: ...
    ↓
Step 100: SAVE CHECKPOINT ← atomic write
    ↓
Step 101-199: ...
    ↓
GPU FAILURE / PREEMPTION (can happen at any time)
    ↓
New GPU allocated (spot instance replacement)
    ↓
LOAD LATEST CHECKPOINT (step 100)
    ↓
Resume from step 100 (lose only steps 101-199)
    ↓
Continue training...
    ↓
Training complete
```

---

## 3. Implementation: Checkpointing with HuggingFace Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",
    
    # Checkpoint frequency
    save_strategy="steps",
    save_steps=50,              # checkpoint every 50 steps
    save_total_limit=3,         # keep only 3 most recent (saves storage)
    
    # Best model tracking
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Evaluation frequency (for best_model tracking)
    evaluation_strategy="steps",
    eval_steps=50,
    
    # Resume capability
    resume_from_checkpoint=True,  # auto-detect latest checkpoint in output_dir
)

# The Trainer saves:
# checkpoint-50/
#   config.json
#   trainer_state.json          ← step, epoch, best metric
#   optimizer.pt                ← AdamW state
#   scheduler.pt                ← LR scheduler state
#   rng_state.pth               ← Python + NumPy + CUDA RNG
#   adapter_model.safetensors   ← LoRA weights
#   adapter_config.json
```

### Resuming training:
```python
# Automatic resume from latest checkpoint
trainer.train(resume_from_checkpoint=True)

# Or explicit checkpoint:
trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-150")
```

---

## 4. Manual Checkpoint Implementation

For custom training loops (e.g., when using GRPO or custom RL loops):

```python
import torch
import os
from pathlib import Path
import random
import numpy as np

def save_checkpoint(
    step: int,
    model,
    optimizer,
    scheduler,
    loss: float,
    output_dir: str,
    max_checkpoints: int = 3,
):
    """Save a complete, resumable checkpoint."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    
    # ATOMIC WRITE: write to temp first, then rename
    # This prevents corrupted checkpoints if process is killed during write
    tmp_dir = Path(output_dir) / f"checkpoint-{step}.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model weights (LoRA adapters or full model)
    model.save_pretrained(tmp_dir)
    
    # 2. Optimizer state
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    
    # 3. LR scheduler state
    torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
    
    # 4. RNG state (all sources)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "cpu": torch.get_rng_state(),
    }
    torch.save(rng_state, tmp_dir / "rng_state.pt")
    
    # 5. Training metadata
    metadata = {
        "step": step,
        "loss": loss,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    import json
    with open(tmp_dir / "trainer_state.json", "w") as f:
        json.dump(metadata, f)
    
    # Atomic rename: if this succeeds, checkpoint is complete
    tmp_dir.rename(checkpoint_dir)
    print(f"Checkpoint saved: {checkpoint_dir}")
    
    # Cleanup old checkpoints
    all_checkpoints = sorted(
        [d for d in Path(output_dir).iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    for old_ckpt in all_checkpoints[:-max_checkpoints]:
        import shutil
        shutil.rmtree(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")


def load_checkpoint(checkpoint_dir: str, model, optimizer, scheduler):
    """Resume training from a checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # 1. Load model weights
    model.load_adapter(checkpoint_dir)  # for LoRA
    # Or: model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))
    
    # 2. Load optimizer state
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
    
    # 3. Load scheduler state
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt"))
    
    # 4. Restore RNG state
    rng_state = torch.load(checkpoint_dir / "rng_state.pt")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.cuda.set_rng_state_all(rng_state["cuda"])
    torch.set_rng_state(rng_state["cpu"])
    
    # 5. Get resume step
    import json
    with open(checkpoint_dir / "trainer_state.json") as f:
        metadata = json.load(f)
    
    return metadata["step"]


def find_latest_checkpoint(output_dir: str):
    """Find the latest valid checkpoint in output_dir."""
    output_dir = Path(output_dir)
    checkpoints = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-") and not d.name.endswith(".tmp")
    ]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda d: int(d.name.split("-")[1])))
```

---

## 5. Atomic Writes — Why This Matters

**The problem:** If the process is killed during a checkpoint write, you get a PARTIAL checkpoint that looks valid but is corrupted.

```python
# BAD: non-atomic write
model.save_pretrained("./checkpoint-100")
# If killed here, checkpoint-100 exists but optimizer.pt is missing!
torch.save(optimizer.state_dict(), "./checkpoint-100/optimizer.pt")

# GOOD: atomic write (write to temp, then rename)
model.save_pretrained("./checkpoint-100.tmp")
torch.save(optimizer.state_dict(), "./checkpoint-100.tmp/optimizer.pt")
os.rename("./checkpoint-100.tmp", "./checkpoint-100")
# os.rename is atomic on most filesystems — either checkpoint-100 exists fully or not at all
```

**Note:** `os.rename` is only atomic on the same filesystem. On distributed filesystems (NFS) or across filesystem boundaries, use a file-lock or marker file approach.

---

## 6. Checkpoint Retention Policy

Storage costs money. Plan your retention:

```
DURING ACTIVE TRAINING:
  Save: every 50-100 steps
  Keep: last 3 checkpoints (rolling)
  Delete: older checkpoints automatically

FINAL CHECKPOINTS:
  Keep: best checkpoint (by eval loss)
  Keep: final checkpoint (last step)
  Keep: major epoch checkpoints (epoch 1, 2, 3)

LONG-TERM STORAGE:
  Archive best checkpoint to cold storage (S3 Glacier)
  Delete all others after project is done

IMPLEMENTATION:
  TrainingArguments(save_total_limit=3)  ← keeps last 3 automatically
  AND: separately track best checkpoint path
```

---

## 7. Multi-GPU Checkpointing

With FSDP/ZeRO, each GPU holds only part of the model. Saving requires gathering.

```python
# With FSDP: save full model from rank 0
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                           FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    state_dict = model.state_dict()

if rank == 0:  # only rank 0 writes the checkpoint
    torch.save(state_dict, "model_full.pt")

# With DeepSpeed:
# model_engine.save_checkpoint("./checkpoint-100")
# (handles distributed checkpoint automatically)
```

**SHARDED checkpoints:** Instead of gathering to rank 0, each GPU saves its own shard.
Faster (parallel writes) but must load on the same GPU configuration.

---

## 8. Cloud Object Storage for Checkpoints

```python
import boto3
from pathlib import Path

def upload_checkpoint_to_s3(local_path: str, s3_bucket: str, s3_prefix: str):
    """Upload checkpoint directory to S3 for durability."""
    s3 = boto3.client("s3")
    for file_path in Path(local_path).rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.name}"
            s3.upload_file(str(file_path), s3_bucket, s3_key)
            print(f"Uploaded: {s3_key}")

# In training loop:
if step % checkpoint_steps == 0:
    save_checkpoint(step, ...)
    upload_checkpoint_to_s3(
        f"./checkpoints/checkpoint-{step}",
        s3_bucket="my-training-bucket",
        s3_prefix=f"experiments/run-{run_id}/checkpoint-{step}"
    )
```

**Why cloud storage:** Spot instances are terminated and disks are wiped. If your checkpoint is only on local disk, it's gone. Upload to S3/GCS after each save.

---

## 9. Asynchronous Checkpointing

Saving a 14GB model checkpoint blocks training for 20-60 seconds. Asynchronous checkpointing writes in a background thread:

```python
import threading

class AsyncCheckpointSaver:
    def __init__(self):
        self._thread = None
    
    def save_async(self, step, model, optimizer, scheduler, output_dir):
        """Snapshot model state and save in background."""
        # Copy model state (fast — just Python dict copy of tensors)
        state = {
            "step": step,
            "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        # Wait for previous save to finish
        if self._thread and self._thread.is_alive():
            self._thread.join()
        # Start new save in background
        self._thread = threading.Thread(
            target=self._write, args=(state, output_dir, step)
        )
        self._thread.start()
    
    def _write(self, state, output_dir, step):
        """Actual disk write (runs in background)."""
        # write state to disk...
        pass
```

PyTorch's `torch.distributed.checkpoint` and libraries like `TorchSnapshot` provide production-grade async checkpointing.

---

## 10. Interview Questions

**Q1 (Intermediate): Why is saving only model weights insufficient for training recovery?**
Strong: "Because training is stateful in more ways than just weights. Adam's momentum (m) and variance (v) represent the 'recent gradient history' — without them, Adam restarts cold and the first 100+ steps after resume have incorrect weight updates. The LR scheduler has a position in its cosine decay curve — without it, LR resets to the initial high value and you may overshoot. Without RNG state, data shuffling changes and training is no longer reproducible. A complete checkpoint is model weights + optimizer state + scheduler state + RNG state + step number."

**Q2 (Intermediate): A training run is interrupted at step 850. Your checkpoints are at steps 800, 700, 600 (last 3 kept). How do you resume?**
Strong: "Load checkpoint-800 and resume from step 800. Steps 801-850 must be re-run. The data loader will be seeded correctly from the RNG state in checkpoint-800, so it will replay the same data as the original run for steps 801-850. If the training is deterministic (same data order, fixed seed, gradient checkpointing stable), the loss curve will be identical to the original run from step 800 onward."

**Q3 (Advanced): How do you handle checkpointing on spot instances?**
Strong: "Three-layer strategy: First, save locally every 50 steps with atomic writes (write to .tmp, then rename). Second, upload to S3 after each local save — even if the instance is terminated, the checkpoint persists in S3. Third, on startup, check S3 for the latest checkpoint and download if local disk is empty. For the handler: register a SIGTERM handler (spot instance sends SIGTERM 2 minutes before termination) to trigger an immediate checkpoint save and upload before the instance is terminated."

---

## MUST KNOW Summary
- Complete checkpoint = model weights + optimizer (m, v) + scheduler + RNG state + step
- Optimizer state is critical: without it, Adam resets and first 100+ steps are wrong
- Atomic writes: write to .tmp, then rename (prevents corrupted checkpoints)
- keep last 3 checkpoints (rolling) + best checkpoint permanently
- Upload to cloud storage (S3/GCS) — spot instances wipe local disk on termination
- Multi-GPU: each GPU saves its shard, OR rank 0 gathers full model
- SIGTERM handler: save immediately when spot instance signals termination
