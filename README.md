BFSI Voice Emotion Detection System

A voice emotion detection system for BFSI call centers, designed for real Indian telephone audio.

## What it does

Analyzes borrower speech in real-time during debt collection calls to:
- Detect emotional state (calm / frustrated / disengaged)
- Trigger automatic human handoff when distress persists
- Generate empathetic, emotion-aware bot responses
- Respond in natural Indian English voice

## Pipeline
```
Borrower audio
      ↓
Whisper small         — Speech to text
      ↓
wav2vec2 (fine-tuned) — Emotion detection
      ↓
Escalation trigger    — Distress > 60% x 3 chunks → human handoff
      ↓
Llama-3.2-3B          — Emotion-aware response generation
      ↓
Edge TTS              — Indian English voice output (en-IN-NeerjaNeural)
```

## Emotion Classes

| Class | Meaning | Bot Behavior |
|---|---|---|
| Calm | Borrower is cooperative | Direct, professional tone |
| Frustrated | Borrower is escalating | Empathetic, acknowledge first |
| Disengaged | Borrower checked out | Re-engagement questions |

## Escalation Rule

Distress confidence > 60% for 3 consecutive 6-second chunks → escalation flag for human agent handoff.

## Quick Start

### Step 1 — Get the dataset

Shemo is a licensed academic dataset. Download it from Kaggle:

**https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database**

After downloading, place files in:
```
data/raw/shemo/male/
data/raw/shemo/female/
```

Then run preprocessing:
```bash
python data/preprocess.py --shemo_root data/raw/shemo --output_csv data/shemo.csv
```

### Step 2 — Train the emotion model

Open the fine-tuning notebook on Kaggle:

**models/training.ipynb**

- Add your Shemo dataset as input
- Enable GPU T4 x2
- Run all cells top to bottom
- Model saves to `/kaggle/working/wav2vec2-shemo-bfsi`
- Download the saved model folder to your machine

### Step 3 — Run the demo

Open **demo/demo.ipynb** in Google Colab:
- Upload your saved model folder to Google Drive
- Add your Hugging Face token (needed for Llama-3.2-3B access)
- Run all cells
- Upload any `.wav` file and click Analyze and Respond

### Step 4 — Use in your own code
```python
from inference.pipeline import Pipeline

pipeline = Pipeline(
    emotion_model_path="path/to/wav2vec2-shemo-bfsi",
    hf_token="your_hf_token",
)

result = pipeline.run("call_audio.wav")
print(result["emotion"])       # frustrated
print(result["escalate"])      # True / False
print(result["bot_response"])  # empathetic response text
```

## Repo Structure
```
Voice---Emotion-Detection/
├── README.md
├── requirements.txt
├── data/
│   ├── preprocess.py        — Shemo preprocessing + BFSI label remapping
│   └── download_data.py     — Dataset download instructions
├── models/
│   └── training.ipynb       — wav2vec2 fine-tuning on Shemo (run on Kaggle)
├── inference/
│   └── pipeline.py          — Complete production pipeline
├── demo/
│   └── demo.ipynb           — Interactive Colab demo
└── results/
    └── classification_report.txt
```

## Dataset

**Shemo — Persian Speech Emotion Detection Database**
- Download: https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database
- Paper: https://arxiv.org/abs/1906.01155
- 3000 samples, remapped to 3 BFSI classes (calm / frustrated / disengaged)
- Augmented with 8kHz codec simulation + Gaussian noise at 12dB SNR

## Tech Stack

| Component | Model |
|---|---|
| Speech to text | openai/whisper-small |
| Emotion detection | facebook/wav2vec2-base (fine-tuned on Shemo) |
| Language model | meta-llama/Llama-3.2-3B-Instruct (4-bit quantized) |
| Text to speech | Microsoft Edge TTS — en-IN-NeerjaNeural |

## Training Details

- Base model: facebook/wav2vec2-base
- Frozen layers: CNN feature extractor + first 9 transformer layers
- Trainable parameters: 21M out of 90M total
- Training samples: 2136 (85% split)
- Validation samples: 377 (15% split)
- Class weights: balanced (disengaged boosted 3.7x)
- Hardware: Kaggle T4 x2 GPU

## Why This Matters for BFSI

Standard emotion detectors are trained on Western studio-recorded speech.
This system is trained on South Asian speech patterns and augmented to handle
real phone call audio quality — matching the actual conditions of an  call center.

The escalation trigger transforms a passive emotion detector into an active
human handoff decision system — the actual business problem being solved.



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
