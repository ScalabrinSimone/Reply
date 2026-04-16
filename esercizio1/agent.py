import os
import json
import re
import ulid
import pandas as pd
from strands import Agent
from strands.models.openai import OpenAIModel
from langfuse import Langfuse, observe

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
# Langfuse v3-style come nel tutorial "Resource Management":
# - from langfuse import Langfuse, observe
# - @observe(as_type="generation") sul wrapper che chiama l'agente
# - langfuse_client.update_current_trace(session_id=...)
# - langfuse_client.update_current_generation(..., usage_details={...})
# ---------------------------------------------------------------------------
if LANGFUSE_PUBLIC_KEY:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
if LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
if LANGFUSE_HOST:
    os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY or os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=LANGFUSE_SECRET_KEY or os.getenv("LANGFUSE_SECRET_KEY"),
    host=LANGFUSE_HOST or os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    """Genera un session ID univoco: {TEAM_NAME}-{ULID}.

    Allineato al tutorial ufficiale:
    - TEAM_NAME preso da env (default "tutorial")
    - spazi sostituiti da trattini
    """
    team_name = os.getenv("TEAM_NAME", "tutorial")
    team_name = team_name.replace(" ", "-")
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

PROCEDURA OBBLIGATORIA (segui questi passi in ordine):
1. Chiama una volta detect_anomalous_transactions() SENZA parametri per ottenere una prima lista di transazioni candidate sospette basata su:
   - importo anomalo per utente (z-score),
   - orario notturno (00:00-06:00),
   - saldo residuo critico dopo la transazione.

2. Recupera il profilo comportamentale di OGNI utente presente nel dataset:
   - per ciascun user_id chiama get_user_transaction_stats(user_id=...).
   - usa questi dati per capire qual e' il comportamento normale dell'utente (importo medio, deviazione standard, tipi piu' frequenti, fasce orarie tipiche).

3. Analizza il rischio derivante da SMS e mail per OGNI utente:
   - per ciascun user_id chiama analyze_communications(user_id=...).
   - se risk_score >= 40 considera l'utente ad alto rischio online;
   - se risk_score >= 80 considera l'utente estremamente vulnerabile: qualsiasi transazione strana per lui va trattata con grande sospetto.

4. Per le transazioni candidate piu' sospette (ad esempio importo molto anomalo o utente ad alto rischio):
   - chiama check_geo_anomaly(user_id, transaction_id, timestamp, location, ...) per verificare se il GPS nelle ultime 24h e' molto lontano dalla citta' di residenza.
   - se anomaly_flag e' True, considera questo un forte segnale di frode.

5. Per OGNI transazione candidata costruisci un punteggio di rischio qualitativo combinando:
   - segnali di importo/orario/saldo da detect_anomalous_transactions,
   - scostamento dalle abitudini dell'utente da get_user_transaction_stats,
   - livello di rischio comunicazioni da analyze_communications,
   - eventuali anomalie geografiche da check_geo_anomaly.
   Dai piu' peso alle combinazioni di piu' segnali (es. importo anomalo + orario notturno + utente con molti SMS di phishing).

6. Considera fraudolente tutte le transazioni con rischio medio-alto, soprattutto se:
   - l'utente ha ricevuto messaggi di phishing, O
   - il GPS e' molto lontano dalla residenza, O
   - il saldo viene quasi azzerato, O
   - l'importo e' nettamente fuori scala per quell'utente.

Linee guida decisionali:
- Il costo di un FALSO NEGATIVO (frode non rilevata) e' MOLTO piu' alto del costo di un falso positivo.
- In caso di dubbio tra includere o escludere una transazione sospetta, scegli di INCLUDERLA.
- Non filtrare troppo: e' meglio riportare piu' transazioni sospette che perderne alcune molto gravi.

OUTPUT FINALE:
- Restituisci SOLO gli UUID delle transazioni che ritieni potenzialmente fraudolente, uno per riga, senza testo aggiuntivo.

Formato output finale (SOLO questo, nient'altro):
<transaction_id_1>
<transaction_id_2>
...
"""


@observe(as_type="generation")
def run_agent_with_trace(session_id: str, model_id: str, agent: Agent, user_prompt: str) -> str:
    """Esegue l'agente con tracing Langfuse.

    Pattern identico al tutorial Resource Management (versione Strands):
    - @observe(as_type="generation") crea una generation per ogni chiamata
    - update_current_trace(session_id=...) lega tutte le trace al session ID
    - update_current_generation(..., usage_details={...}) riporta i token
    """
    # 1) lega la trace corrente al session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # 2) registra input e modello
    langfuse_client.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": user_prompt[:1000]}],
    )

    # 3) esegui Strands agent
    result = agent(user_prompt)
    output_str = str(result)

    # 4) estrai usage per questa chiamata
    invocation = getattr(getattr(result, "metrics", None), "latest_agent_invocation", None)
    if invocation is not None and hasattr(invocation, "usage"):
        usage = invocation.usage
    else:
        usage = getattr(getattr(result, "metrics", None), "accumulated_usage", {}) or {}

    # 5) aggiorna la generation con output e token
    langfuse_client.update_current_generation(
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

    # Il prompt non include piu' dump JSON lunghi: i dettagli vanno letti SOLO via tool
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
Hai accesso a un dataset condiviso (non incluso nel testo del prompt) con:
- {len(transactions)} transazioni (transaction_id, sender_id, amount, balance_after, transaction_type, timestamp, location, ...)
- {len(data['locations']) if isinstance(data['locations'], list) else 'N'} record di posizione GPS
- {len(data['sms']) if isinstance(data['sms'], list) else 'N'} SMS
- {len(data['mails']) if isinstance(data['mails'], list) else 'N'} email
- {len(data['users']) if isinstance(data['users'], list) else 'N'} profili utente

Utenti presenti nel dataset: {user_ids}

NON fare deduzioni sui dati dal testo del prompt: per leggere i dati usa SEMPRE i tool forniti.
Segui rigorosamente la procedura descritta nel system prompt per:
- ottenere le transazioni candidate sospette,
- analizzare gli utenti a rischio,
- verificare segnali da comunicazioni e geolocalizzazione,
- combinare i segnali e decidere quali transazioni sono potenzialmente fraudolente.

Ricorda: falso negativo = frode non rilevata = penalita' alta. In caso di dubbio, INCLUDI.

Rispondi SOLO con la lista degli UUID delle transazioni fraudolente, uno per riga, senza testo aggiuntivo.
"""

    # Esegui l'agente con tracing Langfuse
    raw_output = run_agent_with_trace(session_id, MODEL_ID, agent, user_prompt)

    # Assicura che tutte le trace siano spedite a Langfuse (non bloccare se fallisce)
    try:
        langfuse_client.flush()
    except Exception:
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
