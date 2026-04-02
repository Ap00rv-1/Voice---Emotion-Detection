# BFSI Voice Emotion Detection System

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
Whisper small       — Speech to text
      ↓
wav2vec2 (fine-tuned) — Emotion detection
      ↓
Escalation trigger  — Distress > 60% × 3 chunks → human handoff
      ↓
Llama-3.2-3B        — Emotion-aware response generation
      ↓
Edge TTS            — Indian English voice output
```

## Emotion Classes

| Class | Meaning | Bot Behavior |
|---|---|---|
| Calm | Borrower is cooperative | Direct, professional tone |
| Frustrated | Borrower is escalating | Empathetic, acknowledge first |
| Disengaged | Borrower checked out | Re-engagement questions |

## Escalation Rule

If distress confidence > 60% for 3 consecutive 6-second chunks → output escalation flag for human agent handoff.

## Dataset

- **Shemo** — Persian emotional speech corpus, remapped to 3 BFSI-relevant classes
- **Augmentation** — 8kHz codec simulation, Gaussian noise at 12dB SNR (phone call degradation)
- **Base model** — facebook/wav2vec2-base, fine-tuned with frozen feature extractor

## Repo Structure
```
arrowhead-voice-emotion/
├── README.md
├── data/
│   └── preprocess.py        — Shemo preprocessing + BFSI label remapping
├── models/
│   └── training.ipynb       — wav2vec2 fine-tuning notebook
├── inference/
│   └── pipeline.py          — Complete inference pipeline
├── demo/
│   └── demo.ipynb           — Interactive Colab demo
└── results/
    └── classification_report.txt
```

## Quick Start
```python
from inference.pipeline import ArrowheadPipeline

pipeline = ArrowheadPipeline(
    emotion_model_path="models/wav2vec2-shemo-bfsi",
    hf_token="your_hf_token",
)

result = pipeline.run("call_audio.wav")
print(result["emotion"])      # frustrated
print(result["escalate"])     # True/False
print(result["bot_response"]) # empathetic response text
```

## Tech Stack

| Component | Model |
|---|---|
| Speech to text | openai/whisper-small |
| Emotion detection | facebook/wav2vec2-base (fine-tuned) |
| Language model | meta-llama/Llama-3.2-3B-Instruct |
| Text to speech | Microsoft Edge TTS — en-IN-NeerjaNeural |

## Why This Matters for BFSI

Standard emotion detectors are trained on Western studio-recorded speech.
This system is trained on South Asian speech patterns and augmented to handle
real phone call audio quality — the actual conditions of an Arrowhead call center.
