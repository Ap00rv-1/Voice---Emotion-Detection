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
Escalation trigger    — Distress > 60% × 3 chunks → human handoff
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
```python
from inference.pipeline import ArrowheadPipeline

pipeline = ArrowheadPipeline(
    emotion_model_path="wav2vec2-shemo-bfsi",
    hf_token="your_hf_token",
)

result = pipeline.run("call_audio.wav")
print(result["emotion"])       # frustrated
print(result["escalate"])      # True/False
print(result["bot_response"])  # empathetic response text
```

## Pre-trained Model

The fine-tuned wav2vec2 emotion model is available on Hugging Face:

**https://huggingface.co/your-hf-username/wav2vec2-shemo-bfsi**
```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model     = Wav2Vec2ForSequenceClassification.from_pretrained("your-hf-username/wav2vec2-shemo-bfsi")
extractor = Wav2Vec2FeatureExtractor.from_pretrained("your-hf-username/wav2vec2-shemo-bfsi")
```

## Dataset

**Shemo — Persian Speech Emotion Detection Database**
- Download: https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database
- Paper: https://arxiv.org/abs/1906.01155
- 3000 samples, remapped to 3 BFSI classes (calm / frustrated / disengaged)
- Augmented with 8kHz codec simulation + Gaussian noise to match phone call quality

After downloading, run:
```bash
python data/preprocess.py --shemo_root data/raw/shemo --output_csv data/shemo.csv
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
│   └── training.ipynb       — wav2vec2 fine-tuning on Shemo
├── inference/
│   └── pipeline.py          — Complete production pipeline
├── demo/
│   └── demo.ipynb           — Interactive Colab demo
└── results/
    └── classification_report.txt
```

## Tech Stack

| Component | Model |
|---|---|
| Speech to text | openai/whisper-small |
| Emotion detection | facebook/wav2vec2-base (fine-tuned on Shemo) |
| Language model | meta-llama/Llama-3.2-3B-Instruct (4-bit) |
| Text to speech | Microsoft Edge TTS — en-IN-NeerjaNeural |

## Training Details

- Base model: facebook/wav2vec2-base
- Frozen layers: CNN feature extractor + first 9 transformer layers
- Trainable parameters: 21M out of 90M total
- Training data: 2136 samples (85% split)
- Validation data: 377 samples (15% split)
- Class weights: balanced (disengaged 3.7x due to imbalance)
- Hardware: Kaggle T4 x2 GPU

## Why This Matters for BFSI

Standard emotion detectors are trained on Western studio-recorded speech.
This system is trained on South Asian speech patterns and augmented to handle
real phone call audio quality — the actual conditions of an Arrowhead call center.

The escalation trigger transforms a passive emotion detector into an active
human handoff decision system — the actual business problem being solved.
