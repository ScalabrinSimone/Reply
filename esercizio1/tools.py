import pandas as pd
from strands import tool
from math import radians, cos, sin, asin, sqrt

# ---------------------------------------------------------------------------
# Dataset condiviso — popolato da agent.py prima di lanciare l'agente.
# Evita di passare JSON grezzo nel prompt (troncamento, context overflow).
# ---------------------------------------------------------------------------
_SHARED_DATA: dict = {}

def set_shared_data(data: dict):
    """Chiamato da agent.py dopo load_all() per rendere i dataset accessibili ai tool."""
    global _SHARED_DATA
    _SHARED_DATA = data


# ---------------------------------------------------------------------------
# Utility interna: distanza Haversine tra due coordinate GPS
# ---------------------------------------------------------------------------
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Restituisce la distanza in km tra due punti GPS."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Tool 1: Statistiche per utente sulle transazioni
# ---------------------------------------------------------------------------
@tool
def get_user_transaction_stats(user_id: str, transactions_json: str = "") -> str:
    """
    Dato un user_id, restituisce statistiche di base sulle sue transazioni:
    importo medio, deviazione standard, ore di attivita' tipiche, tipi usati.
    Utile per identificare comportamenti anomali rispetto alla baseline.
    Legge dal dataset condiviso se transactions_json non e' fornito.
    """
    import json
    if transactions_json:
        txs = pd.DataFrame(json.loads(transactions_json))
    else:
        txs = _SHARED_DATA.get("transactions", pd.DataFrame())
        if not isinstance(txs, pd.DataFrame):
            txs = pd.DataFrame(txs)

    if txs.empty:
        return json.dumps({"error": "Nessuna transazione disponibile"})

    if "timestamp" in txs.columns:
        txs["timestamp"] = pd.to_datetime(txs["timestamp"], errors="coerce")

    # Colonna sender_id (normalizzata da data_loader)
    user_txs = txs[txs["sender_id"] == user_id].copy()
    if user_txs.empty:
        return json.dumps({"error": f"Nessuna transazione trovata per sender {user_id}"})

    stats = {
        "user_id": user_id,
        "n_transactions": len(user_txs),
        "amount_mean": round(user_txs["amount"].mean(), 2),
        "amount_std": round(user_txs["amount"].std(), 2),
        "amount_max": round(user_txs["amount"].max(), 2),
        "transaction_types": user_txs["transaction_type"].value_counts().to_dict(),
        "typical_hours": user_txs["timestamp"].dt.hour.value_counts().head(5).to_dict(),
    }
    return json.dumps(stats, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Controllo anomalia geografica
# ---------------------------------------------------------------------------
@tool
def check_geo_anomaly(user_id: str, transaction_id: str,
                      tx_timestamp: str, tx_location: str,
                      locations_json: str = "", users_json: str = "") -> str:
    """
    Verifica se una transazione avviene in una posizione geografica anomala
    rispetto alla residenza dell'utente e alle sue posizioni GPS recenti.
    Restituisce distanza dalla residenza e flag di anomalia.
    Legge dal dataset condiviso se i parametri JSON non sono forniti.
    """
    import json

    if users_json:
        users = json.loads(users_json)
    else:
        users = _SHARED_DATA.get("users", [])

    if locations_json:
        locations = json.loads(locations_json)
    else:
        locations = _SHARED_DATA.get("locations", [])

    # Trova utente per iban o id
    user = next((u for u in users if u.get("iban") == user_id
                 or str(u.get("id", "")) == user_id
                 or str(u.get("user_id", "")) == user_id), None)
    if not user:
        return json.dumps({"error": f"Utente {user_id} non trovato"})

    home_lat = user.get("lat")
    home_lng = user.get("lng")

    # Chiave di matching per le location: prova biotag, user_id, iban
    user_biotag = user.get("biotag")
    user_iban = user.get("iban")
    tx_time = pd.to_datetime(tx_timestamp, errors="coerce")

    recent_positions = []
    for loc in locations:
        # Matching flessibile: biotag, user_id oppure iban
        loc_matches = (
            (user_biotag and loc.get("biotag") == user_biotag)
            or loc.get("user_id") == user_id
            or loc.get("iban") == user_iban
        )
        if not loc_matches:
            continue
        loc_time = pd.to_datetime(loc.get("datetime", "") or loc.get("timestamp", ""), errors="coerce")
        if pd.notna(loc_time) and pd.notna(tx_time):
            delta_hours = abs((tx_time - loc_time).total_seconds() / 3600)
            if delta_hours <= 24:
                recent_positions.append(loc)

    result = {
        "user_id": user_id,
        "transaction_id": transaction_id,
        "home_city": user.get("residence_city", user.get("city", "?")),
        "home_lat": home_lat,
        "home_lng": home_lng,
        "tx_location": tx_location,
        "recent_gps_positions": len(recent_positions),
        "anomaly_flag": False,
        "notes": []
    }

    if home_lat and home_lng and recent_positions:
        for pos in recent_positions:
            dist = _haversine(home_lat, home_lng,
                              pos.get("lat", home_lat),
                              pos.get("lng", home_lng))
            if dist > 500:
                result["anomaly_flag"] = True
                result["notes"].append(
                    f"GPS a {round(dist)} km dalla residenza nelle 24h precedenti"
                )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: Analisi segnali da SMS/mail (phishing, urgenza, account compromesso)
# ---------------------------------------------------------------------------
@tool
def analyze_communications(user_id: str, sms_json: str = "", mails_json: str = "") -> str:
    """
    Analizza SMS e mail dell'utente cercando segnali di compromissione:
    - messaggi di phishing (falsi PayPal, Amazon, banca)
    - urgenza di verifica account
    - link sospetti (domini typosquatted)
    Restituisce un punteggio di rischio e i segnali trovati.
    Legge dal dataset condiviso se i parametri JSON non sono forniti.
    """
    import json
    import re

    if sms_json:
        sms_data = json.loads(sms_json)
    else:
        sms_data = _SHARED_DATA.get("sms", [])

    if mails_json:
        mail_data = json.loads(mails_json)
    else:
        mail_data = _SHARED_DATA.get("mails", [])

    PHISHING_PATTERNS = [
        r"amaz0n", r"paypa1", r"verify.*account", r"suspicious.*login",
        r"account.*lock", r"urgent", r"verify.*identity", r"prevent.*lock",
        r"secure.*account", r"suspicious.*sign.?in", r"click.*link",
        r"confirm.*password", r"reset.*password", r"unusual.*activity",
    ]

    risk_signals = []

    # SMS
    user_sms = []
    if isinstance(sms_data, list):
        user_sms = [s for s in sms_data
                    if str(s.get("user_id", "")) == user_id
                    or str(s.get("recipient", "")) == user_id]
    elif isinstance(sms_data, dict):
        user_sms = sms_data.get(user_id, [])

    for msg in user_sms:
        text = str(
            msg.get("sms", "")
            or msg.get("text", "")
            or msg.get("content", "")
        ).lower()
        for pattern in PHISHING_PATTERNS:
            if re.search(pattern, text):
                risk_signals.append(
                    f"SMS sospetto: '{pattern}' — '{text[:80]}'"
                )
                break

    # Mail
    user_mails = []
    if isinstance(mail_data, list):
        user_mails = [m for m in mail_data
                      if str(m.get("user_id", "")) == user_id
                      or str(m.get("recipient", "")) == user_id]
    elif isinstance(mail_data, dict):
        user_mails = mail_data.get(user_id, [])

    for mail in user_mails:
        text = str(
            mail.get("mail", "")
            or mail.get("body", "")
            or mail.get("content", "")
        ).lower()
        for pattern in PHISHING_PATTERNS:
            if re.search(pattern, text):
                risk_signals.append(
                    f"Mail sospetta: '{pattern}' — '{text[:80]}'"
                )
                break

    risk_score = min(len(risk_signals) * 20, 100)

    return json.dumps({
        "user_id": user_id,
        "risk_score": risk_score,
        "n_suspicious_messages": len(risk_signals),
        "signals": risk_signals[:10]
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: Rilevamento transazioni anomale per importo/ora/tipo
# ---------------------------------------------------------------------------
@tool
def detect_anomalous_transactions(transactions_json: str = "", z_threshold: float = 2.2) -> str:
    """
    Identifica transazioni anomale per:
    - importo statisticamente anomalo rispetto alla media dell'utente (z-score)
    - orario notturno (00:00 - 06:00)
    - saldo negativo o quasi azzerato dopo la transazione (< 50)
    - tipo di transazione insolito per quell'utente (raramente usato, usato solo come segnale secondario)
    Restituisce lista di transaction_id sospetti con motivazione.
    Legge dal dataset condiviso se transactions_json non e' fornito.
    """
    import json

    if transactions_json:
        txs = pd.DataFrame(json.loads(transactions_json))
    else:
        txs = _SHARED_DATA.get("transactions", pd.DataFrame())
        if not isinstance(txs, pd.DataFrame):
            txs = pd.DataFrame(txs)

    if txs.empty:
        return json.dumps([])

    txs = txs.copy()
    txs["timestamp"] = pd.to_datetime(txs["timestamp"], errors="coerce")
    txs["hour"] = txs["timestamp"].dt.hour
    txs["amount"] = pd.to_numeric(txs["amount"], errors="coerce")
    txs["balance_after"] = pd.to_numeric(txs["balance_after"], errors="coerce")

    suspicious = []

    # Z-score per importo per ogni utente (colonna normalizzata: sender_id)
    user_stats = txs.groupby("sender_id")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["sender_id", "mean_amount", "std_amount"]
    txs = txs.merge(user_stats, on="sender_id", how="left")
    txs["z_score"] = (txs["amount"] - txs["mean_amount"]) / txs["std_amount"].replace(0, 1)

    # Frequenze dei tipi di transazione per utente (per rilevare tipi rari)
    type_counts = txs.groupby(["sender_id", "transaction_type"]).size().reset_index(name="count")
    user_counts = txs.groupby("sender_id").size().reset_index(name="total")
    type_freq = type_counts.merge(user_counts, on="sender_id")
    type_freq["freq"] = type_freq["count"] / type_freq["total"].replace(0, 1)

    txs = txs.merge(
        type_freq[["sender_id", "transaction_type", "freq"]],
        on=["sender_id", "transaction_type"],
        how="left",
    )

    for _, row in txs.iterrows():
        reasons = []
        tid = row.get("transaction_id", "")
        if not tid:
            continue

        # Importo anomalo (z-score piu' conservativo per ridurre falsi positivi)
        if pd.notna(row["z_score"]) and row["z_score"] > z_threshold:
            reasons.append(f"importo anomalo (z={round(row['z_score'], 2)})")

        # Orario notturno esteso: 00:00-06:00
        if pd.notna(row["hour"]) and 0 <= row["hour"] < 6:
            reasons.append(f"orario notturno ({int(row['hour'])}:xx)")

        # Saldo residuo critico
        if pd.notna(row["balance_after"]) and row["balance_after"] < 50:
            reasons.append(f"saldo residuo critico ({row['balance_after']})")

        # Tipo di transazione raro per questo utente (freq < 10%): solo come segnale aggiuntivo
        freq = row.get("freq")
        if pd.notna(freq) and freq < 0.1 and reasons:
            reasons.append("tipo di transazione raro per questo utente")

        if reasons:
            suspicious.append({
                "transaction_id": tid,
                "sender_id": row.get("sender_id", ""),
                "amount": row.get("amount", ""),
                "hour": int(row["hour"]) if pd.notna(row["hour"]) else None,
                "balance_after": row.get("balance_after", ""),
                "transaction_type": row.get("transaction_type", ""),
                "reasons": reasons,
            })

    return json.dumps(suspicious, ensure_ascii=False)
