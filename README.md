# Voice Emotion Detection for AI Call Agents

Real-time caller emotion detection system designed for voice AI platforms
operating at Indian telephony scale.

## The Problem

Most voice AI platforms know **what** a caller says but not **how** they feel.
Without emotion awareness, AI agents can't:
- Detect when a caller is about to hang up
- Know when to switch from automated to human handling
- Adapt tone based on caller frustration or disengagement

This system adds that missing layer.

## What it does

- Detects caller emotional state in real-time (calm / frustrated / disengaged)
- Triggers automatic human handoff when distress persists across chunks
- Generates emotion-aware AI agent responses
- Works on real Indian phone call audio — not Western studio recordings

## Pipeline
Caller audio (Indian telephony)
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

## Emotion Classes

| Class | Meaning | Agent Behavior |
|---|---|---|
| Calm | Caller is engaged and cooperative | Continue automated flow |
| Frustrated | Caller is escalating | Switch to empathetic tone |
| Disengaged | Caller has mentally checked out | Re-engagement or handoff |

## Escalation Rule

Distress confidence > 60% for 3 consecutive 6-second chunks
→ escalation flag for human agent handoff.

This single rule transforms a passive emotion detector into an active
call routing decision system.

## Built for Indian Telephony

Most open-source emotion models fail on Indian phone calls because:
- Trained on Western studio-recorded speech (RAVDESS, TESS)
- Real Indian calls are 8kHz, codec-compressed, with background noise

This system addresses both:

| Problem | Solution |
|---|---|
| Wrong speech patterns | Trained on Shemo — South Asian acoustic profile |
| Clean audio assumption | Augmented with 8kHz codec + 12dB noise during training |
| Wrong emotion classes | Remapped to call-center-relevant classes |

## Integration

Drop into any existing voice AI pipeline with 3 lines:
```python
from inference.pipeline import ArrowheadPipeline

pipeline = ArrowheadPipeline(
    emotion_model_path="path/to/wav2vec2-shemo-bfsi",
    hf_token="your_hf_token",
)

result = pipeline.run("call_audio.wav")

print(result["emotion"])       # calm / frustrated / disengaged
print(result["escalate"])      # True → route to human agent
print(result["bot_response"])  # emotion-aware text response
print(result["audio_output"])  # path to TTS audio file
```

## How to Reproduce

### Step 1 — Get the dataset

Download Shemo from Kaggle:
https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database

Place files in:
data/raw/shemo/male/
data/raw/shemo/female/

Run preprocessing:
```bash
python data/preprocess.py --shemo_root data/raw/shemo --output_csv data/shemo.csv
```

### Step 2 — Train the emotion model

Open models/training.ipynb on Kaggle:
- Add Shemo dataset as input
- Enable GPU T4 x2
- Run all cells — model saves to /kaggle/working/wav2vec2-shemo-bfsi
- Download the saved model folder

### Step 3 — Run the demo

Open demo/demo.ipynb in Google Colab:
- Upload saved model to Google Drive
- Add Hugging Face token (for Llama-3.2-3B access)
- Upload any .wav file and click Analyze and Respond

## Repo Structure
Voice---Emotion-Detection/
├── README.md
├── requirements.txt
├── data/
│   ├── preprocess.py        — Shemo preprocessing + label remapping
│   └── download_data.py     — Dataset download instructions
├── models/
│   └── training.ipynb       — wav2vec2 fine-tuning on Kaggle
├── inference/
│   └── pipeline.py          — Production pipeline
├── demo/
│   └── demo.ipynb           — Interactive Colab demo
└── results/
└── classification_report.txt

## Tech Stack

| Component | Model |
|---|---|
| Speech to text | openai/whisper-small |
| Emotion detection | facebook/wav2vec2-base (fine-tuned) |
| Language model | meta-llama/Llama-3.2-3B-Instruct (4-bit) |
| Text to speech | Microsoft Edge TTS — en-IN-NeerjaNeural |

## Training Details

- Base model: facebook/wav2vec2-base
- Frozen: CNN feature extractor + first 9 transformer layers
- Trainable parameters: 21M of 90M total
- Dataset: 2513 samples (Shemo, BFSI-remapped)
- Hardware: Kaggle T4 x2 GPU

## What's next

- Add Hindi speech data (IIIT-H corpus) for stronger Indian language coverage
- Multilingual emotion detection (Hindi, Tamil, Telugu)
- Streaming inference for real-time chunk processing
- REST API wrapper for direct integration into voice AI stacks
"""
