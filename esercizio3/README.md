# Esercizio 3 — Fraud Detection con Audio (STT)

Estende l'esercizio 1 aggiungendo l'analisi di file audio (voicemail/chiamate)
tramite **faster-whisper** (STT 100% locale, nessuna API esterna).

## Struttura

```
esercizio3/
├── agent.py          # Agente Strands con 5 tool
├── config.py         # Path, chiavi, modello Whisper
├── data_loader.py    # Carica CSV, JSON e lista MP3 dalla cartella audio/
├── main.py           # Entry point
├── tools.py          # 5 tool: detect, stats, comms, audio, geo
├── transcriber.py    # Wrapper faster-whisper (singleton)
├── requirements.txt
└── data/
    ├── transactions-5-5.csv
    ├── locations-2-2.json
    ├── mails-3-3.json
    ├── sms-4-4.json
    ├── users-6-6.json
    └── audio/
        └── *.mp3     # formato: YYYYMMDD_HHMMSS-nome_cognome.mp3
```

## Setup

```bash
cd esercizio3
python -m venv .venv
.venv\Scripts\activate  # Windows
# oppure: source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Copia il file `.env` dalla root del progetto (stesse chiavi dell'esercizio 1).

## Dataset

Copia i file nella cartella `data/` e gli MP3 in `data/audio/`:

| File | Descrizione |
|---|---|
| `transactions-5-5.csv` | Transazioni |
| `locations-2-2.json` | GPS |
| `mails-3-3.json` | Email |
| `sms-4-4.json` | SMS |
| `users-6-6.json` | Profili utente |
| `data/audio/*.mp3` | Voicemail/chiamate |

## Formato nome file audio

```
YYYYMMDD_HHMMSS-nome_cognome.mp3
```

Esempio: `20871105_005758-guido_dohn.mp3`
- Data/ora: 5 novembre 2087 alle 00:57:58
- Utente coinvolto: Guido Dohn

Il `data_loader.py` estrae automaticamente timestamp e nome utente.

## Modello Whisper

Il primo avvio scarica il modello (`small` di default, ~150 MB).  
Per cambiare dimensione: `WHISPER_MODEL=base` (piu' veloce) o `WHISPER_MODEL=medium` (piu' accurato) nel `.env`.

## Esecuzione

```bash
python main.py
```

L'output viene scritto in `output.txt`:
- Prima riga: session ID (es. `myteam-01ARZ3NDEKTSV4RRFFQ69G5FAV`)
- Righe successive: transaction_id fraudolenti
