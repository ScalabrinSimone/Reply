import os
import re
import pandas as pd
import json
from config import (
    TRANSACTIONS_FILE, LOCATIONS_FILE, MAILS_FILE,
    SMS_FILE, USERS_FILE, AUDIO_DIR,
)


def load_transactions() -> pd.DataFrame:
    """Carica le transazioni dal CSV e normalizza i nomi delle colonne."""
    df = pd.read_csv(TRANSACTIONS_FILE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_json(path: str) -> list | dict:
    """Carica un file JSON generico."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_audio_files() -> list[dict]:
    """Elenca i file MP3 nella cartella audio e ne parsifica il nome.

    Formato atteso: YYYYMMDD_HHMMSS-nome_cognome.mp3
    Restituisce una lista di dict con:
      - path:      percorso assoluto/relativo al file
      - filename:  nome file (senza path)
      - timestamp: stringa datetime estratta dal nome (YYYY-MM-DD HH:MM:SS)
      - user_name: 'nome cognome' estratto dal nome file
    """
    pattern = re.compile(
        r"^(\d{8})_(\d{6})-([a-z]+_[a-z]+)\.mp3$", re.IGNORECASE
    )
    files = []
    if not os.path.isdir(AUDIO_DIR):
        print(f"[loader] Cartella audio non trovata: {AUDIO_DIR}")
        return files

    for fname in sorted(os.listdir(AUDIO_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        date_str, time_str, name_raw = m.group(1), m.group(2), m.group(3)
        ts = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        user_name = name_raw.replace("_", " ")
        files.append({
            "path": os.path.join(AUDIO_DIR, fname),
            "filename": fname,
            "timestamp": ts,
            "user_name": user_name,
        })

    print(f"[loader] File audio trovati: {len(files)}")
    return files


def load_all() -> dict:
    """Carica tutti i dataset e li restituisce come dizionario."""
    transactions = load_transactions()
    locations    = load_json(LOCATIONS_FILE)
    mails        = load_json(MAILS_FILE)
    sms          = load_json(SMS_FILE)
    users        = load_json(USERS_FILE)
    audio_files  = list_audio_files()

    print(f"[loader] Transazioni caricate: {len(transactions)}")
    print(f"[loader] Utenti: {len(users) if isinstance(users, list) else 'dict'}")
    print(f"[loader] Location records: {len(locations) if isinstance(locations, list) else 'dict'}")
    print(f"[loader] File audio: {len(audio_files)}")

    return {
        "transactions": transactions,
        "locations":    locations,
        "mails":        mails,
        "sms":          sms,
        "users":        users,
        "audio_files":  audio_files,
    }
