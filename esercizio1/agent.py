import os
import json
import re
import ulid
import pandas as pd
from strands import Agent
from strands.models.openai import OpenAIModel
from langfuse import get_client, observe

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_ID,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    OUTPUT_FILE,
)
from data_loader import load_all
from tools import (
    get_user_transaction_stats,
    check_geo_anomaly,
    analyze_communications,
    detect_anomalous_transactions,
    set_shared_data,
)

# ---------------------------------------------------------------------------
# Strands OpenAIModel legge api_key e base_url dalle env var OPENAI_*
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY or ""
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL or ""

# ---------------------------------------------------------------------------
# Langfuse: usiamo il pattern del tutorial ufficiale "Resource Management".
# - SDK Python con decorator @observe(as_type="generation")
# - client singleton via get_client()
# - update_current_trace(session_id=...) per associare il session id
# - update_current_generation(..., usage_details={...}) per i token
# NOTA: le credenziali nel .env della challenge usano nomi custom
#       (langfuse_publicKey, langfuse_privateKey, langfuse_host).
#       Qui le ributtiamo anche nelle variabili attese da Langfuse
#       (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).
# ---------------------------------------------------------------------------
if LANGFUSE_PUBLIC_KEY:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
if LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
if LANGFUSE_HOST:
    os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)

langfuse = get_client()


def generate_session_id() -> str:
    """Genera un session ID univoco: {TEAM_NAME}-{ULID}.
    Il TEAM_NAME nel .env deve avere spazi sostituiti da trattini.
    """
    team_name = os.getenv("TEAM_NAME", "team")
    team_name = team_name.replace(" ", "-")  # sicurezza extra
    return f"{team_name}-{ulid.new().str}"


def build_model() -> OpenAIModel:
    """Costruisce il modello Strands. Credenziali da env var OPENAI_*."""
    return OpenAIModel(
        model_id=MODEL_ID,
        params={
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    )


# ---------------------------------------------------------------------------
# System prompt
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


@observe(as_type="generation")
def run_agent_with_trace(session_id: str, model_id: str, agent: Agent, user_prompt: str) -> str:
    """Esegue l'agente con tracing Langfuse.

    Pattern preso dal tutorial ufficiale della challenge (sessionid.txt):
    - @observe(as_type="generation") crea una generation Langfuse per ogni chiamata
    - langfuse.update_current_trace(session_id=...) associa il session id alla trace
    - langfuse.update_current_generation(..., usage_details={...}) invia i token
    """
    # Associa il session_id alla trace corrente
    langfuse.update_current_trace(session_id=session_id)

    # Registra input e modello
    langfuse.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": user_prompt[:1000]}],
    )

    # Esegui Strands agent
    result = agent(user_prompt)
    output_str = str(result)

    # Estrai usage dall'ultima invocazione dell'agente
    invocation = getattr(getattr(result, "metrics", None), "latest_agent_invocation", None)
    usage = {}
    if invocation is not None and hasattr(invocation, "usage"):
        usage = invocation.usage
    elif hasattr(result, "metrics") and hasattr(result.metrics, "accumulated_usage"):
        usage = result.metrics.accumulated_usage

    # Aggiorna la generation con output e token
    langfuse.update_current_generation(
        model=model_id,
        output=output_str[:1000],
        usage_details={
            "input": usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total": usage.get("totalTokens", 0),
        },
    )

    return output_str


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def run_fraud_detection():
    print("[agent] Caricamento dataset...")
    data = load_all()
    transactions: pd.DataFrame = data["transactions"]

    # Rende il dataset accessibile ai tool senza passarlo nel prompt
    set_shared_data(data)

    tx_col = "transaction_id"
    user_ids = transactions["sender_id"].dropna().unique().tolist()
    users_json = json.dumps(data["users"])
    sms_preview = json.dumps(data["sms"])[:2000]
    mails_preview = json.dumps(data["mails"])[:2000]
    loc_preview = json.dumps(data["locations"])[:1000]

    session_id = generate_session_id()
    print(f"[agent] Session ID: {session_id}")
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
- {len(transactions)} transazioni (colonne: transaction_id, sender_id, amount, balance_after, transaction_type, timestamp)
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

    # Esegui l'agente con tracing Langfuse
    raw_output = run_agent_with_trace(session_id, MODEL_ID, agent, user_prompt)

    # Assicura che tutte le trace siano spedite a Langfuse
    try:
        langfuse.flush()
    except Exception:
        # Non bloccare la gara in caso di problemi di rete con Langfuse
        pass

    # Estrai UUID validi dall'output del modello
    uuids = re.findall(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        raw_output,
        re.IGNORECASE,
    )

    # Filtra solo ID che esistono nel dataset reale
    valid_ids = set(transactions[tx_col].astype(str).tolist())
    fraud_ids = list(dict.fromkeys(uid for uid in uuids if uid in valid_ids))

    # Output: session_id prima riga, poi un transaction_id per riga
    with open(OUTPUT_FILE, "w") as f:
        f.write(session_id + "\n")
        f.write("\n".join(fraud_ids))

    print(f"\n[agent] Trovate {len(fraud_ids)} transazioni fraudolente.")
    print(f"[agent] Session ID: {session_id}")
    print(f"[agent] Output: {OUTPUT_FILE}")
    print(f"[agent] Langfuse dashboard: {LANGFUSE_HOST}")

    print("\n=== TRANSAZIONI FRAUDOLENTE ===")
    for fid in fraud_ids:
        print(fid)

    return fraud_ids


if __name__ == "__main__":
    run_fraud_detection()
