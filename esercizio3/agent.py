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
    analyze_audio_calls,
    set_shared_data,
)

# ---------------------------------------------------------------------------
# Credenziali Strands/OpenRouter
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"]  = OPENROUTER_API_KEY or ""
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL or ""

# ---------------------------------------------------------------------------
# Langfuse v3-style
# ---------------------------------------------------------------------------
for key, val in [
    ("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY),
    ("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY),
    ("LANGFUSE_HOST",       LANGFUSE_HOST),
]:
    if val:
        os.environ.setdefault(key, val)

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY or os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=LANGFUSE_SECRET_KEY or os.getenv("LANGFUSE_SECRET_KEY"),
    host=LANGFUSE_HOST or os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    team_name = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team_name}-{ulid.new().str}"


def build_model() -> OpenAIModel:
    return OpenAIModel(
        model_id=MODEL_ID,
        params={"temperature": 0.1, "max_tokens": 4096},
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Sei un agente specializzato nel rilevamento di frodi finanziarie per MirrorPay nel 2087.
Hai accesso a cinque strumenti:
  1. detect_anomalous_transactions  — rileva transazioni anomale per importo/orario/saldo
  2. get_user_transaction_stats     — profilo comportamentale di un utente
  3. analyze_communications         — analisi SMS e mail per phishing
  4. analyze_audio_calls            — trascrizione STT e analisi segnali di frode nelle chiamate vocali
  5. check_geo_anomaly              — verifica anomalia geografica GPS

PROCEDURA OBBLIGATORIA:
1. Chiama detect_anomalous_transactions() senza parametri per la lista iniziale di sospetti.
2. Per OGNI utente nel dataset:
   a. get_user_transaction_stats(user_id=<id>)
   b. analyze_communications(user_id=<id>)
   c. analyze_audio_calls(user_name=<nome_cognome>) — usa il nome in chiaro, es. 'guido dohn'
3. Per le transazioni piu' sospette chiama check_geo_anomaly(...).
4. Combina TUTTI i segnali per ogni transazione:
   - z-score importo, orario notturno, saldo critico
   - risk_score comunicazioni (SMS/mail)
   - risk_score chiamate vocali (se l'utente ha audio con segnali di vishing/frode)
   - anomalia geografica GPS
5. Considera fraudolente le transazioni con combinazione di piu' segnali,
   o con un segnale molto forte (es. vishing evidente + importo anomalo).

Linee guida:
- Falso negativo (frode non rilevata) = penalita' alta.
- In caso di dubbio tra includere/escludere, scegli di INCLUDERE.
- Restituisci SOLO gli UUID delle transazioni fraudolente, uno per riga.

Formato output finale (SOLO questo):
<transaction_id_1>
<transaction_id_2>
...
"""


@observe(as_type="generation")
def run_agent_with_trace(session_id: str, model_id: str, agent: Agent, user_prompt: str) -> str:
    """Esegue l'agente con tracing Langfuse (pattern Resource Management)."""
    langfuse_client.update_current_trace(session_id=session_id)
    langfuse_client.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": user_prompt[:1000]}],
    )

    result     = agent(user_prompt)
    output_str = str(result)

    invocation = getattr(getattr(result, "metrics", None), "latest_agent_invocation", None)
    usage = (
        invocation.usage
        if invocation is not None and hasattr(invocation, "usage")
        else getattr(getattr(result, "metrics", None), "accumulated_usage", {}) or {}
    )
    langfuse_client.update_current_generation(
        model=model_id,
        output=output_str[:1000],
        usage_details={
            "input":  usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total":  usage.get("totalTokens", 0),
        },
    )
    return output_str


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def run_fraud_detection():
    print("[agent] Caricamento dataset (inclusi audio)...")
    data = load_all()
    transactions: pd.DataFrame = data["transactions"]
    audio_files: list = data.get("audio_files", [])

    # Rende tutto il dataset accessibile ai tool senza passarlo nel prompt
    set_shared_data(data)

    tx_col   = "transaction_id"
    user_ids = transactions["sender_id"].dropna().unique().tolist()

    session_id = generate_session_id()
    print(f"[agent] Session ID: {session_id}")
    print(f"[agent] Transazioni: {len(transactions)} | Utenti: {len(user_ids)} | Audio: {len(audio_files)}")

    model = build_model()
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            detect_anomalous_transactions,
            get_user_transaction_stats,
            analyze_communications,
            analyze_audio_calls,
            check_geo_anomaly,
        ],
    )

    # Nomi utenti dagli audio (es. 'guido dohn') per il prompt
    audio_users = list({af["user_name"] for af in audio_files})

    user_prompt = f"""
Hai accesso a un dataset condiviso con:
- {len(transactions)} transazioni
- {len(data['locations']) if isinstance(data['locations'], list) else 'N'} record GPS
- {len(data['sms']) if isinstance(data['sms'], list) else 'N'} SMS
- {len(data['mails']) if isinstance(data['mails'], list) else 'N'} email
- {len(data['users']) if isinstance(data['users'], list) else 'N'} profili utente
- {len(audio_files)} file audio (voicemail/chiamate)

Utenti nel dataset transazioni: {user_ids}
Utenti con file audio: {audio_users}

LeggI i dati SOLO tramite i tool, non dai testi del prompt.
Segui la procedura descritta nel system prompt.
Ricorda: falso negativo = penalita' alta. In dubbio, INCLUDI.

Rispondi SOLO con la lista degli UUID delle transazioni fraudolente, uno per riga.
"""

    raw_output = run_agent_with_trace(session_id, MODEL_ID, agent, user_prompt)

    try:
        langfuse_client.flush()
    except Exception:
        pass

    uuids = re.findall(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        raw_output, re.IGNORECASE,
    )
    valid_ids  = set(transactions[tx_col].astype(str).tolist())
    fraud_ids  = list(dict.fromkeys(uid for uid in uuids if uid in valid_ids))

    with open(OUTPUT_FILE, "w") as f:
        f.write(session_id + "\n")
        f.write("\n".join(fraud_ids))

    print(f"\n[agent] Trovate {len(fraud_ids)} transazioni fraudolente.")
    print(f"[agent] Session ID: {session_id}")
    print(f"[agent] Output: {OUTPUT_FILE}")

    print("\n=== TRANSAZIONI FRAUDOLENTE ===")
    for fid in fraud_ids:
        print(fid)

    return fraud_ids


if __name__ == "__main__":
    run_fraud_detection()
