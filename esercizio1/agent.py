import os
import json
import re
import pandas as pd
from strands import Agent
from strands.models.openai import OpenAIModel
from langfuse import get_client, observe

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_ID,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    OUTPUT_FILE
)
from data_loader import load_all
from tools import (
    get_user_transaction_stats,
    check_geo_anomaly,
    analyze_communications,
    detect_anomalous_transactions,
)

# ---------------------------------------------------------------------------
# Langfuse v3: credenziali come env var PRIMA di qualsiasi chiamata
# ---------------------------------------------------------------------------
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY or ""
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY or ""
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST or ""

# ---------------------------------------------------------------------------
# Strands OpenAIModel: api_key e base_url NON sono parametri diretti del costruttore.
# Il modo corretto e' settarli come env var OPENAI_API_KEY e OPENAI_BASE_URL,
# che il client openai sottostante legge automaticamente.
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY or ""
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL or ""


def build_model() -> OpenAIModel:
    """Costruisce il modello Strands. Le credenziali vengono lette da env var."""
    return OpenAIModel(
        model_id=MODEL_ID,
        params={
            "temperature": 0.1,  # bassa temperatura = output deterministico
            "max_tokens": 4096,
        }
    )


# ---------------------------------------------------------------------------
# Prompt di sistema per l'agente
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Sei un agente specializzato nel rilevamento di frodi finanziarie per MirrorPay nel 2087.
Hai accesso a strumenti per analizzare transazioni, posizioni GPS, comunicazioni (SMS, email) e profili utente.

Il tuo obiettivo e' identificare transazioni fraudolente nel dataset fornito.

Principi guida:
1. Una transazione e' sospetta se presenta piu' segnali combinati: importo anomalo, orario notturno,
   posizione geografica incoerente con la residenza, o se l'utente ha ricevuto messaggi di phishing recenti.
2. Preferisci avere qualche falso positivo piuttosto che perdere frodi reali (il costo del falso negativo e' alto).
3. Analizza prima le transazioni anomale per tipo statistico, poi arricchisci con dati geografici e comunicativi.
4. Restituisci SOLO gli ID delle transazioni che ritieni fraudolente, uno per riga, senza testo aggiuntivo.

Formato output finale:
<transaction_id_1>
<transaction_id_2>
...
"""


# ---------------------------------------------------------------------------
# @observe (Langfuse v3): traccia automaticamente input/output/latenza/token
# ---------------------------------------------------------------------------
@observe(name="fraud-detection-esercizio1")
def _run_agent(agent: Agent, user_prompt: str) -> str:
    response = agent(user_prompt)
    return str(response)


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def run_fraud_detection():
    print("[agent] Caricamento dataset...")
    data = load_all()
    transactions: pd.DataFrame = data["transactions"]

    tx_json = transactions.to_json(orient="records", date_format="iso")
    loc_json = json.dumps(data["locations"])
    sms_json = json.dumps(data["sms"])
    users_json = json.dumps(data["users"])

    print(f"[agent] Avvio analisi su {len(transactions)} transazioni...")

    model = build_model()
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_user_transaction_stats,
            check_geo_anomaly,
            analyze_communications,
            detect_anomalous_transactions,
        ],
    )

    user_prompt = f"""
Hai a disposizione i seguenti dataset:
- {len(transactions)} transazioni
- Dati GPS degli utenti
- SMS e email degli utenti
- Profili di {len(data['users']) if isinstance(data['users'], list) else 'N'} utenti

Procedi nell'ordine:
1. detect_anomalous_transactions: individua transazioni sospette per importo, orario, saldo.
2. get_user_transaction_stats: per ogni sospetto, confronta con la baseline dell'utente.
3. analyze_communications: verifica messaggi di phishing sugli utenti coinvolti.
4. check_geo_anomaly: verifica coerenza geografica.
5. Decidi quali transazioni sono fraudolente.

Transactions JSON:
{tx_json[:8000]}

Users JSON:
{users_json}

SMS JSON (prime 3000 char):
{json.dumps(data['sms'])[:3000]}

Locations JSON (prime 2000 char):
{loc_json[:2000]}

Rispondi SOLO con la lista degli UUID delle transazioni fraudolente, uno per riga.
"""

    raw_output = _run_agent(agent, user_prompt)

    # Estrai UUID validi dall'output del modello
    uuids = re.findall(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        raw_output, re.IGNORECASE
    )

    # Filtra solo ID che esistono nel dataset reale
    valid_ids = set(transactions["transactionid"].astype(str).tolist())
    fraud_ids = list(dict.fromkeys(uid for uid in uuids if uid in valid_ids))

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(fraud_ids))

    print(f"\n[agent] Trovate {len(fraud_ids)} transazioni fraudolente.")
    print(f"[agent] Output: {OUTPUT_FILE}")
    print(f"[agent] Langfuse dashboard: {LANGFUSE_HOST}")

    get_client().flush()
    return fraud_ids


if __name__ == "__main__":
    run_fraud_detection()
