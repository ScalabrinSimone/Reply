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
from prescorer import compute_risk_scores
from tools import (
    get_user_transaction_stats,
    check_geo_anomaly,
    analyze_communications,
    analyze_audio_calls,
    detect_anomalous_transactions,
    set_shared_data,
)

os.environ["OPENAI_API_KEY"]  = OPENROUTER_API_KEY or ""
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL or ""

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
        params={"temperature": 0.1, "max_tokens": 8192},
    )


SYSTEM_PROMPT = """
Sei un agente specializzato nel rilevamento di frodi finanziarie per MirrorPay nel 2087.

Ricevi una lista di transazioni candidate con risk_score pre-calcolato (0-100)
e le motivazioni dei segnali (inclusi eventuali segnali da chiamate vocali/vishing,
IBAN nuovi, spike di frequenza, importi alti).

Hai a disposizione tool per approfondire:
  - get_user_transaction_stats(user_id)           -> profilo comportamentale utente
  - analyze_communications(user_id)               -> analisi phishing SMS/mail
  - analyze_audio_calls(user_name)                -> trascrizione STT e segnali vishing
  - check_geo_anomaly(user_id, tx_id, ...)        -> verifica GPS
  - detect_anomalous_transactions()               -> lista completa anomalie statistiche

REGOLE DI DECISIONE:
  risk_score >= 55  -> INCLUDI sempre (segnali forti)
  risk_score 30-54  -> INCLUDI se almeno un tool conferma un segnale
  risk_score 10-29  -> INCLUDI solo se il tool rivela un pattern chiaro di frode
                       oppure se l'importo e' alto (>=3000) con almeno 1 segnale

CRITERIO ECONOMICO:
  Dai priorita' alle transazioni con importo piu' elevato: una frode da 5000 ha
  impatto economico molto maggiore di una da 50. Se risk_reasons contiene
  'importo alto assoluto' e ci sono altri segnali, INCLUDI.

CRITERIO FONDAMENTALE:
  Falso negativo (frode non rilevata) >> costo falso positivo.
  In caso di dubbio, INCLUDI.
  Segnale 'vishing audio' o 'IBAN mai visto' = forte indicatore, tratta l'utente come ad alto rischio.

OUTPUT: SOLO UUID delle transazioni fraudolente, uno per riga. Nessun testo aggiuntivo.
"""


@observe(as_type="generation")
def run_agent_with_trace(session_id: str, model_id: str, agent: Agent, user_prompt: str) -> str:
    langfuse_client.update_current_trace(session_id=session_id)
    langfuse_client.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": user_prompt[:2000]}],
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
        output=output_str[:2000],
        usage_details={
            "input":  usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total":  usage.get("totalTokens", 0),
        },
    )
    return output_str


def run_fraud_detection():
    print("[agent] Caricamento dataset (inclusi audio)...")
    data = load_all()
    transactions: pd.DataFrame = data["transactions"]
    audio_files: list = data.get("audio_files", [])
    set_shared_data(data)

    session_id = generate_session_id()
    print(f"[agent] Session ID: {session_id}")
    print(f"[agent] Transazioni: {len(transactions)} | Audio: {len(audio_files)}")

    print("[agent] Pre-scoring deterministico v2...")
    scored = compute_risk_scores(data)
    print(f"[agent] Candidate (score>=10): {len(scored)}")

    # Stampa distribuzione score per debug
    if not scored.empty:
        bins = [10, 20, 30, 40, 55, 101]
        labels = ["10-19", "20-29", "30-39", "40-54", "55+"]
        scored["bucket"] = pd.cut(scored["risk_score"], bins=bins, labels=labels, right=False)
        print("[agent] Distribuzione score:", scored["bucket"].value_counts().to_dict())

    candidates_json = scored[[
        "transaction_id", "sender_id", "amount", "hour",
        "balance_after", "transaction_type", "z_score",
        "risk_score", "risk_reasons",
    ]].to_dict(orient="records")

    # Limite contesto: max 250 candidate (claude-opus ha contesto largo)
    MAX_CANDIDATES = 250
    if len(candidates_json) > MAX_CANDIDATES:
        print(f"[agent] Troncamento a {MAX_CANDIDATES} candidate.")
        candidates_json = candidates_json[:MAX_CANDIDATES]

    model = build_model()
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_user_transaction_stats,
            check_geo_anomaly,
            analyze_communications,
            analyze_audio_calls,
            detect_anomalous_transactions,
        ],
    )

    user_prompt = f"""Analizza le seguenti {len(candidates_json)} transazioni candidate (ordinate per risk_score decrescente).
Per ogni transazione hai il risk_score (0-100) e le motivazioni.

Regole:
- score >= 55: includi direttamente.
- score 30-54: usa un tool per confermare, poi includi se confermato.
- score 10-29: usa un tool, includi solo se trovi frode evidente o importo alto con segnale.
- Se risk_reasons include 'vishing audio' o 'IBAN mai visto': tratta come alta priorita'.
- Se 'importo alto assoluto' (>=3000) + almeno 1 altro segnale: includi.

Transazioni candidate:
{json.dumps(candidates_json, ensure_ascii=False, indent=None)}

Rispondi SOLO con la lista degli UUID delle transazioni fraudolente, uno per riga."""

    raw_output = run_agent_with_trace(session_id, MODEL_ID, agent, user_prompt)

    try:
        langfuse_client.flush()
    except Exception:
        pass

    uuids = re.findall(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        raw_output, re.IGNORECASE,
    )
    valid_ids = set(transactions["transaction_id"].astype(str).tolist())
    fraud_ids = list(dict.fromkeys(uid for uid in uuids if uid in valid_ids))

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
