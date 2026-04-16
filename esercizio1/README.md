# Esercizio 1 — Fraud Detection Agent

Agente AI basato su **Strands** per il rilevamento di frodi finanziarie nel dataset di training della Reply AI Agent Challenge 2026.

## Setup

### 1. Crea il virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Configura il `.env`
Copia `.env.example` in `.env` e inserisci le chiavi:
```bash
cp .env.example .env
```

### 4. Metti i file dati nella cartella `data/`
```
esercizio1/data/
├── transactions-5.csv
├── locations-2.json
├── mails-3.json
├── sms-4.json
└── users-6.json
```

### 5. Esegui
```bash
python main.py
```

L'output sara' scritto in `output.txt`, uno UUID per riga.

## Architettura

```
main.py          # Entrypoint
agent.py         # Logica agente Strands + Langfuse
tools.py         # Tool custom (@tool decorator)
data_loader.py   # Caricamento e parsing dataset
config.py        # Variabili d'ambiente e costanti
```

## Tool disponibili

| Tool | Scopo |
|------|-------|
| `detect_anomalous_transactions` | Z-score su importo, orario notturno, saldo critico |
| `get_user_transaction_stats` | Baseline comportamentale per utente |
| `analyze_communications` | Pattern phishing in SMS/email |
| `check_geo_anomaly` | Coerenza GPS con residenza utente |

## Monitoraggio costi
I costi e i token usati sono visibili su Langfuse:
`https://challenges.reply.com/langfuse`

## Modello usato
`anthropic/claude-3-5-haiku` via OpenRouter — buon rapporto qualita'/costo per task analitici strutturati.
