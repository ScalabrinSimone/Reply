import os
import json
import pandas as pd
from strands import Agent
from strands.models.openai import OpenAIModel
from langfuse import Langfuse

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
# Setup Langfuse per tracciare token e costi
# ---------------------------------------------------------------------------
def setup_langfuse() -> Langfuse:
    lf = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    return lf


# ---------------------------------------------------------------------------
# Setup modello via OpenRouter
# ---------------------------------------------------------------------------
def build_model() -> OpenAIModel:
    """
    Usa OpenAIModel di Strands puntando a OpenRouter come base_url.
    claude-3-5-haiku: ottimo per task analitici, molto economico.
    """
    return OpenAIModel(
        model_id=MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        params={
            "temperature": 0.1,  # bassa temperatura = output piu' deterministico
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
# Funzione principale
# ---------------------------------------------------------------------------
def run_fraud_detection():
    print("[agent] Caricamento dataset...")
    data = load_all()
    transactions: pd.DataFrame = data["transactions"]

    # Serializza i dataset come JSON string per passarli ai tool
    tx_json = transactions.to_json(orient="records", date_format="iso")
    loc_json = json.dumps(data["locations"])
    sms_json = json.dumps(data["sms"])
    mail_json = json.dumps(data["mails"])
    users_json = json.dumps(data["users"])

    # Setup Langfuse
    langfuse = setup_langfuse()
    trace = langfuse.trace(
        name="fraud-detection-esercizio1",
        metadata={"n_transactions": len(transactions)}
    )

    print(f"[agent] Avvio analisi su {len(transactions)} transazioni...")

    # Costruisce l'agente con tutti i tool
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

    # Prompt utente: descrizione del task con i dati inline
    # Passiamo un riassunto dei dati per non saturare il contesto
    user_prompt = f"""
Hai a disposizione i seguenti dataset (gia' caricati nei tool):
- {len(transactions)} transazioni (transactions_json disponibile nei tool)
- Dati GPS di localizzazione degli utenti
- SMS e email degli utenti
- Profili di {len(data['users']) if isinstance(data['users'], list) else 'N'} utenti

Usa i tool disponibili per:
1. detect_anomalous_transactions: individua transazioni sospette per importo, orario, saldo.
2. Per ogni transazione sospetta trovata, usa get_user_transaction_stats per verificare se e' anomala rispetto alla baseline dell'utente.
3. analyze_communications: verifica se gli utenti coinvolti hanno ricevuto messaggi di phishing.
4. check_geo_anomaly: verifica coerenza geografica per transazioni in-person o con location specificata.
5. Sulla base di tutti i segnali raccolti, decidi quali transazioni sono fraudolente.

Transactions JSON (da passare ai tool):
{tx_json[:8000]}...  [troncato per contesto, usa il tool detect_anomalous_transactions]

Users JSON:
{users_json}

SMS JSON (prime 3000 char):
{sms_json[:3000]}

Locations JSON (prime 2000 char):
{loc_json[:2000]}

Rispondi SOLO con la lista degli ID delle transazioni fraudolente, uno per riga.
"""

    # Esegui l'agente
    span = trace.span(name="agent-run")
    response = agent(user_prompt)
    span.end()

    # Estrai gli ID dall'output (righe che sembrano UUID)
    import re
    raw_output = str(response)
    uuids = re.findall(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        raw_output, re.IGNORECASE
    )

    # Verifica che gli ID trovati esistano nel dataset
    valid_ids = set(transactions["transactionid"].astype(str).tolist())
    fraud_ids = [uid for uid in uuids if uid in valid_ids]
    fraud_ids = list(dict.fromkeys(fraud_ids))  # deduplica mantenendo ordine

    # Scrivi output
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(fraud_ids))

    print(f"\n[agent] Trovate {len(fraud_ids)} transazioni fraudolente sospette.")
    print(f"[agent] Output scritto in: {OUTPUT_FILE}")
    print("[agent] Controlla i costi su Langfuse:", LANGFUSE_HOST)

    langfuse.flush()
    return fraud_ids


if __name__ == "__main__":
    run_fraud_detection()
