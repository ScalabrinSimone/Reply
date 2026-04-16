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
    set_shared_data,  # rende i dataset accessibili ai tool senza passarli nel prompt
)

# ---------------------------------------------------------------------------
# Langfuse v3: credenziali come env var PRIMA di qualsiasi chiamata
# ---------------------------------------------------------------------------
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY or ""
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY or ""
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST or ""

# ---------------------------------------------------------------------------
# Strands OpenAIModel legge api_key e base_url dalle env var OPENAI_*
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY or ""
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL or ""


def build_model() -> OpenAIModel:
    """Costruisce il modello Strands. Le credenziali vengono lette da env var."""
    return OpenAIModel(
        model_id=MODEL_ID,
        params={
            "temperature": 0.1,
            "max_tokens": 4096,
        }
    )


# ---------------------------------------------------------------------------
# Prompt di sistema per l'agente
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Sei un agente specializzato nel rilevamento di frodi finanziarie per MirrorPay nel 2087.
Hai accesso a strumenti per analizzare transazioni, posizioni GPS, comunicazioni (SMS, email) e profili utente.

Il tuo obiettivo e' identificare TUTTE le transazioni fraudolente nel dataset. E' fondamentale non perderne nessuna.

REGOLE OBBLIGATORIE:
1. Devi chiamare TUTTI e 4 i tool prima di dare una risposta finale:
   - detect_anomalous_transactions (senza parametri: legge dal dataset condiviso)
   - get_user_transaction_stats per OGNI utente presente nel dataset
   - analyze_communications per OGNI utente presente nel dataset
   - check_geo_anomaly per le transazioni sospette
2. Una transazione e' fraudolenta se presenta ALMENO UNO di questi segnali:
   - importo statisticamente anomalo (z-score > 2.0)
   - orario notturno (00:00-06:00)
   - saldo residuo critico dopo la transazione (< 50)
   - utente ha ricevuto SMS o mail di phishing
   - anomalia geografica (GPS lontano dalla residenza)
3. Il costo di un FALSO NEGATIVO (frode non rilevata) e' MOLTO PIU' ALTO del costo di un falso positivo.
   Quindi: in caso di dubbio, INCLUDI la transazione nella lista.
4. Non filtrare troppo: e' meglio riportare 20 transazioni sospette che perderne 5.
5. Restituisci SOLO gli UUID delle transazioni fraudolente, uno per riga, senza testo aggiuntivo.

Formato output finale (SOLO questo, nient'altro):
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

    # Rende i dataset accessibili ai tool senza passarli nel prompt
    # (evita troncamento JSON e context overflow)
    set_shared_data(data)

    tx_col = "transaction_id"  # colonna normalizzata da data_loader

    # Estrai user_id univoci per il prompt
    user_ids = transactions["sender_id"].dropna().unique().tolist()
    users_json = json.dumps(data["users"])

    # Riassunti leggeri per contestualizzare (non JSON grezzo delle tx)
    sms_preview = json.dumps(data["sms"])[:2000]
    mails_preview = json.dumps(data["mails"])[:2000]
    loc_preview = json.dumps(data["locations"])[:1000]

    print(f"[agent] Avvio analisi su {len(transactions)} transazioni...")
    print(f"[agent] Utenti univoci: {user_ids}")

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
Dataset disponibile (caricato nel contesto condiviso, accessibile dai tool):
- {len(transactions)} transazioni (colonne normalizzate: transaction_id, sender_id, amount, balance_after, transaction_type, timestamp)
- {len(data['locations']) if isinstance(data['locations'], list) else 'N'} record GPS
- SMS e mail degli utenti
- {len(data['users']) if isinstance(data['users'], list) else 'N'} profili utente

Utenti presenti: {user_ids}

Istruzioni:
1. Chiama detect_anomalous_transactions (senza parametri) per avere la lista iniziale di sospetti.
2. Per ciascuno degli utenti {user_ids}, chiama get_user_transaction_stats(user_id=<id>).
3. Per ciascuno degli utenti {user_ids}, chiama analyze_communications(user_id=<id>).
4. Per le transazioni piu' sospette, chiama check_geo_anomaly.
5. Combina tutti i segnali e produci la lista finale.

Ricorda: falso negativo = frode non rilevata = penalita' alta. In caso di dubbio, includi.

Profili utente (JSON):
{users_json}

Anteprima SMS:
{sms_preview}

Anteprima Mail:
{mails_preview}

Anteprima Locations:
{loc_preview}

Rispondi SOLO con la lista degli UUID delle transazioni fraudolente, uno per riga.
"""

    raw_output = _run_agent(agent, user_prompt)

    # Estrai UUID validi dall'output del modello
    uuids = re.findall(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        raw_output, re.IGNORECASE
    )

    # Filtra solo ID che esistono nel dataset reale
    valid_ids = set(transactions[tx_col].astype(str).tolist())
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
