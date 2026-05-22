"""Microphone recording and speech-to-text via IBM Granite Speech on torch."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from minimlx.defaults import SILENCE_THRESHOLD, SILENCE_DURATION  # noqa: F401

SAMPLE_RATE = 16_000
CHANNELS = 1
MAX_RECORD_SECONDS = 120   # hard cap


def _check_sounddevice() -> None:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        raise SystemExit(
            "sounddevice is required for conversation mode.\n"
            "  pip install 'minimlx[conversation]'"
        )


def record_audio(
    *,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
    on_start: Optional[callable] = None,
) -> np.ndarray:
    """Record from the default microphone until silence or max duration.

    Returns a 1-D float32 numpy array at 16 kHz.
    """
    _check_sounddevice()
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    silent_frames = 0
    chunk_samples = int(SAMPLE_RATE * 0.1)  # 100 ms chunks
    silence_chunks_needed = int(silence_duration / 0.1)
    max_chunks = int(max_seconds / 0.1)
    stop_event = threading.Event()

    def _callback(indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        chunks.append(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < silence_threshold:
            nonlocal silent_frames
            silent_frames += 1
        else:
            silent_frames = 0
        if silent_frames >= silence_chunks_needed or len(chunks) >= max_chunks:
            stop_event.set()

    if on_start:
        on_start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_samples,
        callback=_callback,
    ):
        stop_event.wait(timeout=max_seconds)

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks)
    # Trim trailing silence
    if silent_frames > 0:
        trim = min(silent_frames * chunk_samples, len(audio) - chunk_samples)
        audio = audio[: len(audio) - trim]
    return audio


def record_until_enter(
    *,
    max_seconds: float = MAX_RECORD_SECONDS,
) -> np.ndarray:
    """Record from the default microphone until the user presses Enter.

    Returns a 1-D float32 numpy array at 16 kHz.
    """
    _check_sounddevice()
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    chunk_samples = int(SAMPLE_RATE * 0.1)
    stop_event = threading.Event()

    def _callback(indata, frames, time_info, status):
        chunks.append(indata[:, 0].copy())
        if len(chunks) >= int(max_seconds / 0.1):
            stop_event.set()

    def _wait_enter():
        try:
            sys.stdin.readline()
        except EOFError:
            pass
        stop_event.set()

    t = threading.Thread(target=_wait_enter, daemon=True)
    t.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_samples,
        callback=_callback,
    ):
        stop_event.wait(timeout=max_seconds)

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


class Transcriber:
    """Speech-to-text via IBM Granite 4.0 1B Speech.

    Lazy-loads the model on first call so the import cost is deferred.
    Language is auto-detected by the model — the `language` kwarg is kept
    for CLI compatibility but only used as a keyword-bias hint in the prompt.
    """

    def __init__(self, model_id: str = "ibm-granite/granite-4.0-1b-speech", language: str = "en"):
        self.model_id = model_id
        self.language = language
        self._processor = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        except ImportError:
            raise SystemExit(
                "transformers and torch are required for speech-to-text.\n"
                "  pip install 'minimlx[conversation]'"
            )
        self._torch = torch
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 16 kHz float32 audio array to text."""
        if len(audio) < SAMPLE_RATE * 0.3:
            return ""
        self._load()
        torch = self._torch

        # Granite expects a [1, num_samples] float tensor at 16 kHz, mono.
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)

        prompt_text = "<|audio|>can you transcribe the speech into a written format?"
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._processor(
            prompt, wav, device=self._device, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=440,
                do_sample=False,
                num_beams=1,
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        texts = self._tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return texts[0].strip() if texts else ""
