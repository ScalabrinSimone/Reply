import re
import json
import pandas as pd
from strands import tool
from math import radians, cos, sin, asin, sqrt
from transcriber import transcribe

# ---------------------------------------------------------------------------
# Dataset condiviso — popolato da agent.py prima di lanciare l'agente.
# ---------------------------------------------------------------------------
_SHARED_DATA: dict = {}

def set_shared_data(data: dict):
    """Chiamato da agent.py dopo load_all() per rendere i dataset accessibili ai tool."""
    global _SHARED_DATA
    _SHARED_DATA = data


# ---------------------------------------------------------------------------
# Utility interna: distanza Haversine
# ---------------------------------------------------------------------------
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Tool 1: STT + analisi contenuto audio
# ---------------------------------------------------------------------------
@tool
def analyze_audio_calls(user_name: str = "") -> str:
    """Trascrive tutti i file audio (voicemail/chiamate) relativi a un utente
    (o tutti se user_name e' vuoto) usando faster-whisper in locale.

    Dal nome file estrae:
    - timestamp della chiamata (YYYYMMDD_HHMMSS)
    - nome utente coinvolto

    Dal contenuto trascritto cerca segnali di frode:
    - menzione di trasferimenti urgenti, bonifici non autorizzati
    - richieste di dati bancari, PIN, OTP
    - vishing (finti operatori bancari)
    - pressione temporale ('entro oggi', 'subito', 'urgente')

    Restituisce un JSON con le trascrizioni e il punteggio di rischio per utente.
    """
    FRAUD_PATTERNS = [
        r"urgent", r"immediately", r"transfer", r"bank.*account", r"pin\b",
        r"otp", r"password", r"verify", r"suspend", r"block.*card",
        r"click.*link", r"call.*back", r"refund", r"prize", r"won",
        r"subito", r"urgente", r"bonifico", r"conto", r"sospeso",
        r"verifica", r"codice", r"clicca", r"premio", r"vinto",
        r"urgent", r"virement", r"compte", r"suspendu", r"verification",
    ]

    audio_files = _SHARED_DATA.get("audio_files", [])
    if not audio_files:
        return json.dumps({"error": "Nessun file audio nel dataset condiviso"})

    # Filtra per utente se specificato
    if user_name:
        target = user_name.lower().replace("_", " ")
        audio_files = [
            f for f in audio_files
            if target in f["user_name"].lower()
        ]

    results = []
    for af in audio_files:
        transcript = transcribe(af["path"])
        text_lower = transcript.lower()

        signals = []
        for pattern in FRAUD_PATTERNS:
            if re.search(pattern, text_lower):
                signals.append(pattern)

        risk_score = min(len(signals) * 25, 100)

        results.append({
            "filename":   af["filename"],
            "timestamp":  af["timestamp"],
            "user_name":  af["user_name"],
            "transcript": transcript[:500],   # troncato per non appesantire il contesto
            "risk_signals": signals,
            "risk_score": risk_score,
        })

    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Statistiche per utente sulle transazioni
# ---------------------------------------------------------------------------
@tool
def get_user_transaction_stats(user_id: str, transactions_json: str = "") -> str:
    """Dato un user_id, restituisce statistiche di base sulle sue transazioni:
    importo medio, deviazione standard, ore tipiche, tipi usati.
    Legge dal dataset condiviso se transactions_json non e' fornito.
    """
    import json as _json
    if transactions_json:
        txs = pd.DataFrame(_json.loads(transactions_json))
    else:
        txs = _SHARED_DATA.get("transactions", pd.DataFrame())
        if not isinstance(txs, pd.DataFrame):
            txs = pd.DataFrame(txs)

    if txs.empty:
        return _json.dumps({"error": "Nessuna transazione disponibile"})

    txs["timestamp"] = pd.to_datetime(txs.get("timestamp", pd.Series()), errors="coerce")
    user_txs = txs[txs["sender_id"] == user_id].copy()
    if user_txs.empty:
        return _json.dumps({"error": f"Nessuna transazione per {user_id}"})

    stats = {
        "user_id":          user_id,
        "n_transactions":   len(user_txs),
        "amount_mean":      round(user_txs["amount"].mean(), 2),
        "amount_std":       round(user_txs["amount"].std(), 2),
        "amount_max":       round(user_txs["amount"].max(), 2),
        "transaction_types": user_txs["transaction_type"].value_counts().to_dict(),
        "typical_hours":    user_txs["timestamp"].dt.hour.value_counts().head(5).to_dict(),
    }
    return _json.dumps(stats, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: Anomalie geografiche
# ---------------------------------------------------------------------------
@tool
def check_geo_anomaly(user_id: str, transaction_id: str,
                      tx_timestamp: str, tx_location: str,
                      locations_json: str = "", users_json: str = "") -> str:
    """Verifica se una transazione avviene in una posizione geografica anomala
    rispetto alla residenza dell'utente e alle sue posizioni GPS recenti.
    Legge dal dataset condiviso se i parametri JSON non sono forniti.
    """
    import json as _json

    users     = _json.loads(users_json)     if users_json     else _SHARED_DATA.get("users", [])
    locations = _json.loads(locations_json) if locations_json else _SHARED_DATA.get("locations", [])

    user = next(
        (u for u in users
         if u.get("iban") == user_id
         or str(u.get("id", "")) == user_id
         or str(u.get("user_id", "")) == user_id),
        None,
    )
    if not user:
        return _json.dumps({"error": f"Utente {user_id} non trovato"})

    home_lat  = user.get("lat")
    home_lng  = user.get("lng")
    user_biotag = user.get("biotag")
    user_iban   = user.get("iban")
    tx_time   = pd.to_datetime(tx_timestamp, errors="coerce")

    recent_positions = []
    for loc in locations:
        loc_matches = (
            (user_biotag and loc.get("biotag") == user_biotag)
            or loc.get("user_id") == user_id
            or loc.get("iban") == user_iban
        )
        if not loc_matches:
            continue
        loc_time = pd.to_datetime(
            loc.get("datetime", "") or loc.get("timestamp", ""), errors="coerce"
        )
        if pd.notna(loc_time) and pd.notna(tx_time):
            if abs((tx_time - loc_time).total_seconds() / 3600) <= 24:
                recent_positions.append(loc)

    result = {
        "user_id": user_id, "transaction_id": transaction_id,
        "home_city": user.get("residence_city", user.get("city", "?")),
        "home_lat": home_lat, "home_lng": home_lng,
        "tx_location": tx_location,
        "recent_gps_positions": len(recent_positions),
        "anomaly_flag": False, "notes": [],
    }

    if home_lat and home_lng:
        for pos in recent_positions:
            dist = _haversine(home_lat, home_lng,
                              pos.get("lat", home_lat), pos.get("lng", home_lng))
            if dist > 500:
                result["anomaly_flag"] = True
                result["notes"].append(f"GPS a {round(dist)} km dalla residenza nelle 24h")

    return _json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: Analisi SMS e mail
# ---------------------------------------------------------------------------
@tool
def analyze_communications(user_id: str, sms_json: str = "", mails_json: str = "") -> str:
    """Analizza SMS e mail dell'utente cercando segnali di phishing/compromissione.
    Restituisce risk_score e segnali trovati.
    Legge dal dataset condiviso se i parametri JSON non sono forniti.
    """
    import json as _json

    sms_data  = _json.loads(sms_json)   if sms_json   else _SHARED_DATA.get("sms", [])
    mail_data = _json.loads(mails_json) if mails_json else _SHARED_DATA.get("mails", [])

    PHISHING_PATTERNS = [
        r"amaz0n", r"paypa1", r"verify.*account", r"suspicious.*login",
        r"account.*lock", r"urgent", r"verify.*identity", r"prevent.*lock",
        r"secure.*account", r"suspicious.*sign.?in", r"click.*link",
        r"confirm.*password", r"reset.*password", r"unusual.*activity",
    ]

    risk_signals = []

    def _scan(items, field_keys, source_label):
        user_msgs = [
            m for m in (items if isinstance(items, list) else [])
            if str(m.get("user_id", "")) == user_id
            or str(m.get("recipient", "")) == user_id
        ] or (items.get(user_id, []) if isinstance(items, dict) else [])
        for msg in user_msgs:
            text = ""
            for k in field_keys:
                text = str(msg.get(k, "")).lower()
                if text:
                    break
            for p in PHISHING_PATTERNS:
                if re.search(p, text):
                    risk_signals.append(f"{source_label}: '{p}' — '{text[:80]}'")
                    break

    _scan(sms_data,  ["sms", "text", "content"], "SMS")
    _scan(mail_data, ["mail", "body", "content"], "Mail")

    return _json.dumps({
        "user_id": user_id,
        "risk_score": min(len(risk_signals) * 20, 100),
        "n_suspicious_messages": len(risk_signals),
        "signals": risk_signals[:10],
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 5: Rilevamento transazioni anomale
# ---------------------------------------------------------------------------
@tool
def detect_anomalous_transactions(transactions_json: str = "", z_threshold: float = 2.2) -> str:
    """Identifica transazioni anomale per importo (z-score), orario notturno,
    saldo critico e tipo raro per utente.
    Legge dal dataset condiviso se transactions_json non e' fornito.
    """
    import json as _json

    if transactions_json:
        txs = pd.DataFrame(_json.loads(transactions_json))
    else:
        txs = _SHARED_DATA.get("transactions", pd.DataFrame())
        if not isinstance(txs, pd.DataFrame):
            txs = pd.DataFrame(txs)

    if txs.empty:
        return _json.dumps([])

    txs = txs.copy()
    txs["timestamp"]    = pd.to_datetime(txs["timestamp"], errors="coerce")
    txs["hour"]         = txs["timestamp"].dt.hour
    txs["amount"]       = pd.to_numeric(txs["amount"], errors="coerce")
    txs["balance_after"] = pd.to_numeric(txs["balance_after"], errors="coerce")

    # Z-score per importo per utente
    user_stats = txs.groupby("sender_id")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["sender_id", "mean_amount", "std_amount"]
    txs = txs.merge(user_stats, on="sender_id", how="left")
    txs["z_score"] = (txs["amount"] - txs["mean_amount"]) / txs["std_amount"].replace(0, 1)

    # Frequenza tipo transazione per utente
    type_counts = txs.groupby(["sender_id", "transaction_type"]).size().reset_index(name="count")
    user_counts = txs.groupby("sender_id").size().reset_index(name="total")
    type_freq   = type_counts.merge(user_counts, on="sender_id")
    type_freq["freq"] = type_freq["count"] / type_freq["total"].replace(0, 1)
    txs = txs.merge(type_freq[["sender_id", "transaction_type", "freq"]],
                    on=["sender_id", "transaction_type"], how="left")

    suspicious = []
    for _, row in txs.iterrows():
        reasons = []
        tid = row.get("transaction_id", "")
        if not tid:
            continue
        if pd.notna(row["z_score"]) and row["z_score"] > z_threshold:
            reasons.append(f"importo anomalo (z={round(row['z_score'], 2)})")
        if pd.notna(row["hour"]) and 0 <= row["hour"] < 6:
            reasons.append(f"orario notturno ({int(row['hour'])}:xx)")
        if pd.notna(row["balance_after"]) and row["balance_after"] < 50:
            reasons.append(f"saldo residuo critico ({row['balance_after']})")
        freq = row.get("freq")
        if pd.notna(freq) and freq < 0.1 and reasons:
            reasons.append("tipo di transazione raro per questo utente")
        if reasons:
            suspicious.append({
                "transaction_id":   tid,
                "sender_id":        row.get("sender_id", ""),
                "amount":           row.get("amount", ""),
                "hour":             int(row["hour"]) if pd.notna(row["hour"]) else None,
                "balance_after":    row.get("balance_after", ""),
                "transaction_type": row.get("transaction_type", ""),
                "reasons":          reasons,
            })

    return _json.dumps(suspicious, ensure_ascii=False)
