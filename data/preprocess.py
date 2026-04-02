"""
Shemo Dataset Preprocessing
-----------------------------
Remaps Shemo emotion labels to 3 BFSI-relevant classes
and applies telephone audio augmentation.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path


# Shemo → BFSI label mapping
EMOTION_MAP = {
    "N": "calm",        # Neutral
    "H": "calm",        # Happy — cooperative in BFSI context
    "A": "frustrated",  # Anger
    "F": "disengaged",  # Fear — checked out
    "S": "disengaged",  # Sadness — disengaged
    "W": "frustrated",  # Surprise/worried
}


def simulate_phone_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate telephone codec degradation."""
    # Codec compression: 16kHz → 8kHz → 16kHz
    degraded = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    degraded = librosa.resample(degraded, orig_sr=8000, target_sr=sr)
    # Gaussian noise at 12dB SNR
    signal_power = np.mean(degraded ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (12 / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(degraded))
    degraded     = np.clip(degraded + noise, -1.0, 1.0)
    return degraded.astype(np.float32)


def build_shemo_dataframe(shemo_root: str) -> pd.DataFrame:
    """
    Build a dataframe from Shemo dataset directory.
    Expects structure: shemo_root/male/*.wav and shemo_root/female/*.wav
    Filename format: [SpeakerID][EmotionCode][UtteranceID].wav
    Example: M01A01.wav → Male speaker 01, Anger, utterance 01
    """
    records = []
    for gender_dir in ["male", "female"]:
        dir_path = os.path.join(shemo_root, gender_dir)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith(".wav"):
                continue
            emotion_code = fname[3]   # 4th character is emotion code
            bfsi_label   = EMOTION_MAP.get(emotion_code)
            if bfsi_label is None:
                continue
            records.append({
                "path":    os.path.join(dir_path, fname),
                "emotion": emotion_code,
                "label":   bfsi_label,
                "gender":  gender_dir,
            })

    df = pd.DataFrame(records)
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shemo_root", required=True, help="Path to Shemo dataset root")
    parser.add_argument("--output_csv", default="shemo.csv")
    args = parser.parse_args()

    df = build_shemo_dataframe(args.shemo_root)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")
