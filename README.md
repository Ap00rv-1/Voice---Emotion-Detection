# BFSI Voice Emotion Detection Pipeline — v2 Update

## What's new in this update

The original pipeline fine-tuned `facebook/wav2vec2-base` end-to-end (full fine-tuning) for
3-class emotion classification on BFSI debt-collection call audio. This update adds a
**LoRA (parameter-efficient) fine-tuning variant** and an **inference latency benchmark**,
to evaluate whether the model can be trained more efficiently and whether it's viable for
near-real-time escalation triggering.

## 1. LoRA vs. Full Fine-Tuning

Same dataset, same train/val split (`random_state=42`), same class-weighted loss, same
training hyperparameters (10 epochs, lr=3e-5, warmup 10%) — only the fine-tuning method
changed, to keep the comparison fair.

| Method | Trainable Params | % of Total | Best F1 (weighted) | Best Epoch |
|---|---|---|---|---|
| Full fine-tuning | ~94.5M | 100% | 0.309 | — |
| LoRA (r=8, target: `q_proj`, `v_proj`) | 492,547 | 0.52% | **0.513** | 4 / 10 |

**LoRA outperformed full fine-tuning** on this task despite training ~192x fewer parameters.
The likely explanation: the dataset (Shemo, remapped to 3 business classes, with a class
imbalance skewed against `disengaged`) is small enough that full fine-tuning overfits —
all 94.5M parameters have the capacity to memorize training-set patterns that don't
generalize. LoRA's low-rank update restricts how much the model can shift from its
pretrained representation, which acts as an implicit regularizer here.

Full fine-tuning's F1 also degrades after its peak (see training logs below) — consistent
with overfitting on the minority class as training continues, whereas LoRA's best
checkpoint is preserved via `load_best_model_at_end=True`.

**LoRA config used:**
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier", "projector"],  # new classification head trains fully
)
```

## 2. Inference Latency Benchmark

Measured on the LoRA fine-tuned model, GPU inference, 20 timed runs after 3 warmup runs
(to exclude CUDA kernel compilation overhead), with `torch.cuda.synchronize()` to ensure
accurate wall-clock timing of GPU-async operations.

| Metric | Value |
|---|---|
| Mean latency | 35.5 ms |
| p50 latency | 35.4 ms |
| p95 latency | 37.1 ms |
| Std dev | 1.1 ms |

Latency is tight and predictable (p95 close to p50), which matters more than raw speed
alone for a system triggering escalation in near-real-time during a live call.

## 3. Why this matters for production deployment

- **Training cost**: LoRA fine-tuning is cheaper to iterate on — fewer gradients to
  compute and store, smaller optimizer state, faster experimentation cycles when
  retraining on new call center data.
- **Storage**: LoRA adapter weights are a few MB versus the full ~94.5M-parameter
  checkpoint, making multi-tenant or multi-client model deployment more practical
  (swap adapters per client instead of storing full model copies).
- **Latency**: ~35ms mean inference is well within budget for the 3-consecutive-chunk
  escalation logic in the original pipeline design.

## Reproducing this

The full training notebook (`training.ipynb`) includes both the original full
fine-tuning cell and the LoRA variant. Swap between them by changing the model-setup
cell only — the dataset class, augmentation, class weighting, and trainer logic are
shared and unchanged between both runs.

---
*Original pipeline (Whisper → wav2vec2 → Llama-3.2-3B → Edge TTS) and dataset details
remain as documented in the main README above this section.*
