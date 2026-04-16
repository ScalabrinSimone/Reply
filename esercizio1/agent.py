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
# Langfuse v3: le credenziali vanno settate come env var PRIMA di get_client()
# ---------------------------------------------------------------------------
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY or ""
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY or ""
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST or ""


# ---------------------------------------------------------------------------
# Setup modello via OpenRouter
# ---------------------------------------------------------------------------
def build_model() -> OpenAIModel:
    return OpenAIModel(
        model_id=MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
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
# Funzione agente decorata con @observe (Langfuse v3)
# @observe crea automaticamente uno span/trace per ogni chiamata
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
    mail_json = json.dumps(data["mails"])
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
Hai a disposizione i seguenti dataset (gia' caricati nei tool):
- {len(transactions)} transazioni
- Dati GPS di localizzazione degli utenti
- SMS e email degli utenti
- Profili di {len(data['users']) if isinstance(data['users'], list) else 'N'} utenti

Usa i tool disponibili nell'ordine seguente:
1. detect_anomalous_transactions: passa il JSON delle transazioni e individua quelle sospette per importo, orario, saldo.
2. Per ogni transazione sospetta, usa get_user_transaction_stats per verificare se e' anomala rispetto alla baseline dell'utente.
3. analyze_communications: verifica se gli utenti coinvolti hanno ricevuto messaggi di phishing.
4. check_geo_anomaly: verifica coerenza geografica per transazioni in-person o con location specificata.
5. Decidi quali transazioni sono fraudolente in base ai segnali raccolti.

Transactions JSON (da passare a detect_anomalous_transactions):
{tx_json[:8000]}

Users JSON:
{users_json}

SMS JSON (prime 3000 char):
{sms_json[:3000]}

Locations JSON (prime 2000 char):
{loc_json[:2000]}

Rispondi SOLO con la lista degli ID delle transazioni fraudolente, uno per riga.
"""

    # Esegui agente — @observe traccia input/output/token su Langfuse automaticamente
    raw_output = _run_agent(agent, user_prompt)

    # Estrai UUID validi dall'output
    uuids = re.findall(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        raw_output, re.IGNORECASE
    )

    # Tieni solo ID che esistono realmente nel dataset
    valid_ids = set(transactions["transactionid"].astype(str).tolist())
    fraud_ids = list(dict.fromkeys(uid for uid in uuids if uid in valid_ids))

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(fraud_ids))

    print(f"\n[agent] Trovate {len(fraud_ids)} transazioni fraudolente sospette.")
    print(f"[agent] Output scritto in: {OUTPUT_FILE}")
    print(f"[agent] Controlla token e costi su Langfuse: {LANGFUSE_HOST}")

    # Flush: assicura che tutti gli span vengano inviati a Langfuse
    lf = get_client()
    lf.flush()

    return fraud_ids


if __name__ == "__main__":
    run_fraud_detection()
