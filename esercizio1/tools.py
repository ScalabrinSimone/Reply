import pandas as pd
from strands import tool
from math import radians, cos, sin, asin, sqrt


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
def get_user_transaction_stats(user_id: str, transactions_json: str) -> str:
    """
    Dato un user_id e le transazioni in formato JSON string,
    restituisce statistiche di base: importo medio, deviazione standard,
    ore di attivita' tipiche, tipi di transazione usati.
    Utile per identificare comportamenti anomali rispetto alla baseline.
    """
    import json
    txs = pd.DataFrame(json.loads(transactions_json))
    if "timestamp" in txs.columns:
        txs["timestamp"] = pd.to_datetime(txs["timestamp"], errors="coerce")

    # Filtra per sender
    user_txs = txs[txs["senderid"] == user_id].copy()
    if user_txs.empty:
        return f"Nessuna transazione trovata per sender {user_id}"

    stats = {
        "user_id": user_id,
        "n_transactions": len(user_txs),
        "amount_mean": round(user_txs["amount"].mean(), 2),
        "amount_std": round(user_txs["amount"].std(), 2),
        "amount_max": round(user_txs["amount"].max(), 2),
        "transaction_types": user_txs["transactiontype"].value_counts().to_dict(),
        "typical_hours": user_txs["timestamp"].dt.hour.value_counts().head(5).to_dict(),
    }
    return json.dumps(stats, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Controllo anomalia geografica
# ---------------------------------------------------------------------------
@tool
def check_geo_anomaly(user_id: str, transaction_id: str,
                      tx_timestamp: str, tx_location: str,
                      locations_json: str, users_json: str) -> str:
    """
    Verifica se una transazione avviene in una posizione geografica anomala
    rispetto alla residenza dell'utente e alle sue posizioni GPS recenti.
    Restituisce distanza dalla residenza e flag di anomalia.
    """
    import json
    from datetime import datetime

    users = json.loads(users_json)
    locations = json.loads(locations_json)

    # Trova residenza utente
    user = next((u for u in users if u.get("iban") == user_id
                 or str(u.get("id", "")) == user_id), None)
    if not user:
        return json.dumps({"error": f"Utente {user_id} non trovato"})

    home_lat = user.get("lat")
    home_lng = user.get("lng")

    # Posizioni GPS dell'utente entro 24h prima della transazione
    biotag = user.get("biotag") or user.get("iban")  # fallback
    tx_time = pd.to_datetime(tx_timestamp, errors="coerce")
    recent_positions = []
    for loc in locations:
        if loc.get("biotag") == biotag or loc.get("user_id") == user_id:
            loc_time = pd.to_datetime(loc.get("datetime", ""), errors="coerce")
            if pd.notna(loc_time) and pd.notna(tx_time):
                delta_hours = abs((tx_time - loc_time).total_seconds() / 3600)
                if delta_hours <= 24:
                    recent_positions.append(loc)

    result = {
        "user_id": user_id,
        "transaction_id": transaction_id,
        "home_city": user.get("residence_city", "?"),
        "home_lat": home_lat,
        "home_lng": home_lng,
        "tx_location": tx_location,
        "recent_gps_positions": len(recent_positions),
        "anomaly_flag": False,
        "notes": []
    }

    # Controlla distanza dalla residenza se abbiamo coordinate
    if home_lat and home_lng and recent_positions:
        for pos in recent_positions:
            dist = _haversine(home_lat, home_lng,
                              pos.get("lat", home_lat), pos.get("lng", home_lng))
            if dist > 500:  # > 500 km = anomalia forte
                result["anomaly_flag"] = True
                result["notes"].append(f"GPS a {round(dist)} km dalla residenza nelle 24h precedenti")

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: Analisi segnali da SMS/mail (phishing, urgenza, account compromesso)
# ---------------------------------------------------------------------------
@tool
def analyze_communications(user_id: str, sms_json: str, mails_json: str) -> str:
    """
    Analizza SMS e mail dell'utente cercando segnali di compromissione:
    - messaggi di phishing ricevuti (falsi PayPal, Amazon, banca)
    - urgenza di verifica account
    - link sospetti (domini typosquatted come paypa1, amaz0n)
    Restituisce un punteggio di rischio e i segnali trovati.
    """
    import json
    import re

    sms_data = json.loads(sms_json)
    mail_data = json.loads(mails_json)

    PHISHING_PATTERNS = [
        r"amaz0n", r"paypa1", r"verify.*account", r"suspicious.*login",
        r"account.*lock", r"urgent", r"verify.*identity", r"prevent.*lock",
        r"secure.*account", r"suspicious.*sign.?in"
    ]

    risk_signals = []

    # Cerca nei messaggi SMS dell'utente specifico
    user_sms = []
    if isinstance(sms_data, list):
        user_sms = [s for s in sms_data
                    if str(s.get("user_id", "")) == user_id
                    or str(s.get("recipient", "")) == user_id]
    elif isinstance(sms_data, dict):
        user_sms = sms_data.get(user_id, [])

    for msg in user_sms:
        text = str(msg.get("sms", "") or msg.get("text", "") or msg.get("content", "")).lower()
        for pattern in PHISHING_PATTERNS:
            if re.search(pattern, text):
                risk_signals.append(f"SMS sospetto: pattern '{pattern}'")
                break

    # Cerca nelle mail
    user_mails = []
    if isinstance(mail_data, list):
        user_mails = [m for m in mail_data
                      if str(m.get("user_id", "")) == user_id
                      or str(m.get("recipient", "")) == user_id]
    elif isinstance(mail_data, dict):
        user_mails = mail_data.get(user_id, [])

    for mail in user_mails:
        text = str(mail.get("mail", "") or mail.get("body", "") or mail.get("content", "")).lower()
        for pattern in PHISHING_PATTERNS:
            if re.search(pattern, text):
                risk_signals.append(f"Mail sospetta: pattern '{pattern}'")
                break

    risk_score = min(len(risk_signals) * 20, 100)  # max 100

    return json.dumps({
        "user_id": user_id,
        "risk_score": risk_score,
        "n_suspicious_messages": len(risk_signals),
        "signals": risk_signals[:10]  # max 10 segnali in output
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: Rilevamento transazioni anomale per importo/ora/tipo
# ---------------------------------------------------------------------------
@tool
def detect_anomalous_transactions(transactions_json: str, z_threshold: float = 2.5) -> str:
    """
    Identifica transazioni anomale per:
    - importo statisticamente anomalo rispetto alla media dell'utente (z-score)
    - orario notturno (00:00 - 05:00)
    - saldo negativo o quasi azzerato dopo la transazione
    Restituisce una lista di transaction_id sospetti con motivazione.
    """
    import json
    txs = pd.DataFrame(json.loads(transactions_json))
    if txs.empty:
        return json.dumps([])

    txs["timestamp"] = pd.to_datetime(txs["timestamp"], errors="coerce")
    txs["hour"] = txs["timestamp"].dt.hour
    txs["amount"] = pd.to_numeric(txs["amount"], errors="coerce")
    txs["balanceafter"] = pd.to_numeric(txs["balanceafter"], errors="coerce")

    suspicious = []

    # Calcola z-score per importo per ogni utente
    user_stats = txs.groupby("senderid")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["senderid", "mean_amount", "std_amount"]
    txs = txs.merge(user_stats, on="senderid", how="left")
    txs["z_score"] = (txs["amount"] - txs["mean_amount"]) / txs["std_amount"].replace(0, 1)

    for _, row in txs.iterrows():
        reasons = []
        tid = row.get("transactionid", row.get("transaction_id", ""))
        if not tid:
            continue

        # Importo anomalo (z-score alto)
        if pd.notna(row["z_score"]) and row["z_score"] > z_threshold:
            reasons.append(f"importo anomalo (z={round(row['z_score'], 2)})")

        # Orario notturno
        if pd.notna(row["hour"]) and 0 <= row["hour"] < 5:
            reasons.append(f"orario notturno ({int(row['hour'])}:xx)")

        # Saldo quasi azzerato (< 10% del balance medio stimato)
        if pd.notna(row["balanceafter"]) and row["balanceafter"] < 50:
            reasons.append(f"saldo residuo critico ({row['balanceafter']})")

        if reasons:
            suspicious.append({
                "transaction_id": tid,
                "sender_id": row.get("senderid", ""),
                "amount": row.get("amount", ""),
                "reasons": reasons
            })

    return json.dumps(suspicious, ensure_ascii=False)
