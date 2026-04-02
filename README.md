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
