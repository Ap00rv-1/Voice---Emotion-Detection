"""
Arrowhead BFSI Voice Emotion Pipeline
--------------------------------------
Audio → STT (Whisper) → Emotion (wav2vec2) → Escalation → LLM (Llama-3.2) → TTS (Edge)
"""

import torch
import torchaudio
import numpy as np
import tempfile
import asyncio
import edge_tts
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import whisper

TARGET_SR   = 16000
MAX_SAMPLES = TARGET_SR * 6
ID2LABEL    = {0: "calm", 1: "frustrated", 2: "disengaged"}


class ArrowheadPipeline:
    def __init__(self, emotion_model_path: str, hf_token: str):
        print("Loading emotion model...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(emotion_model_path)
        self.emotion_model     = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_path)
        self.emotion_model.eval().cuda()

        print("Loading Whisper...")
        self.whisper = whisper.load_model("small")

        print("Loading Llama-3.2-3B...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct", token=hf_token
        )
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=bnb,
            device_map="auto",
            token=hf_token,
            low_cpu_mem_usage=True,
        ).eval()

        # Escalation state
        self.distress_buffer = []
        self.history         = []
        print("Pipeline ready")

    def predict_emotion(self, audio_path: str) -> dict:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        audio = waveform.squeeze().numpy()
        audio = audio[:MAX_SAMPLES] if len(audio) > MAX_SAMPLES else np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        inputs = self.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.emotion_model(inputs.input_values.cuda()).logits
        probs   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return {
            "emotion":    ID2LABEL[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "scores":     {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(3)},
        }

    def check_escalation(self, emotion: str, confidence: float) -> bool:
        is_distress = emotion in ["frustrated", "disengaged"] and confidence > 0.60
        self.distress_buffer.append(is_distress)
        if len(self.distress_buffer) > 3:
            self.distress_buffer.pop(0)
        return len(self.distress_buffer) == 3 and all(self.distress_buffer)

    def generate_response(self, transcript: str, emotion: str) -> str:
        system_prompt = f"""You are an empathetic AI assistant for Arrowhead Finance debt collection.
Borrower emotion: {emotion.upper()}
- calm: Be professional, offer repayment options.
- frustrated: Acknowledge frustration first, then discuss payment.
- disengaged: Ask a short open question to re-engage.
Keep response under 3 sentences. Be respectful always."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": transcript})
        input_text = self.llama_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.llama_tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs, max_new_tokens=120, temperature=0.7,
                do_sample=True, pad_token_id=self.llama_tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.llama_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def _tts_async(self, text: str, path: str):
        await edge_tts.Communicate(text=text, voice="en-IN-NeerjaNeural").save(path)

    def text_to_speech(self, text: str) -> str:
        path = tempfile.mktemp(suffix=".wav")
        asyncio.get_event_loop().run_until_complete(self._tts_async(text, path))
        return path

    def run(self, audio_path: str) -> dict:
        # STT
        transcript = self.whisper.transcribe(audio_path)["text"].strip()

        # Emotion
        emotion_result = self.predict_emotion(audio_path)
        emotion        = emotion_result["emotion"]
        confidence     = emotion_result["confidence"]

        # Escalation
        escalate = self.check_escalation(emotion, confidence)

        # LLM
        if escalate:
            bot_text = "I understand this has been difficult. Let me connect you with a senior advisor who can help you right now."
        else:
            bot_text = self.generate_response(transcript, emotion)

        # TTS
        audio_output = self.text_to_speech(bot_text)

        # Update history
        self.history.append({"role": "user",      "content": transcript})
        self.history.append({"role": "assistant",  "content": bot_text})

        return {
            "transcript":   transcript,
            "emotion":      emotion,
            "confidence":   confidence,
            "scores":       emotion_result["scores"],
            "escalate":     escalate,
            "bot_response": bot_text,
            "audio_output": audio_output,
        }

    def reset(self):
        self.distress_buffer = []
        self.history         = []
