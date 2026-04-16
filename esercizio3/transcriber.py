"""Modulo STT con faster-whisper (100% locale, nessuna API esterna).

Installazione necessaria nel venv:
    pip install faster-whisper

Il modello viene scaricato automaticamente al primo utilizzo
nella cache di HuggingFace (~150-500 MB a seconda della size).
"""
from __future__ import annotations

import os
from functools import lru_cache
from config import WHISPER_MODEL_SIZE


@lru_cache(maxsize=1)
def _get_model():
    """Carica il modello faster-whisper una sola volta e lo tiene in cache."""
    from faster_whisper import WhisperModel  # import lazy per non rallentare lo startup

    # device="cpu" + compute_type="int8" = massima compatibilita' senza GPU
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"[STT] Modello faster-whisper '{WHISPER_MODEL_SIZE}' caricato.")
    return model


def transcribe(audio_path: str, language: str = None) -> str:
    """Trascrive un file audio MP3 e restituisce il testo.

    Args:
        audio_path: percorso al file .mp3 (o qualsiasi formato ffmpeg).
        language:   codice lingua ISO-639 (es. 'en', 'it', 'fr').
                    Se None, faster-whisper la rileva automaticamente.

    Returns:
        Testo trascritto come stringa. Stringa vuota se il file non esiste
        o se la trascrizione fallisce.
    """
    if not os.path.isfile(audio_path):
        print(f"[STT] File non trovato: {audio_path}")
        return ""

    try:
        model = _get_model()
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,   # rimuove silenzio iniziale/finale
        )
        detected = info.language
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        print(f"[STT] {os.path.basename(audio_path)} -> lingua={detected}, chars={len(text)}")
        return text
    except Exception as e:
        print(f"[STT] Errore su {audio_path}: {e}")
        return ""
