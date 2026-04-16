"""FIX 2: Pre-scorer deterministico in Python.

Calcola un risk_score numerico per ogni transazione PRIMA di passare
all'agente. Include anche il segnale audio (vishing dalle chiamate).

Risk score per transazione (punti, capped a 100):
  +30  importo anomalo (z-score > 2.2)
  +20  orario notturno (00:00-05:59)
  +25  saldo residuo critico (< 50)
  +15  tipo di transazione raro per utente (freq < 10%, solo se score > 0)
  +20  utente con >=2 messaggi phishing (SMS/mail)
  +10  utente con 1 messaggio phishing
  +25  utente con segnali di vishing negli audio (risk_score audio >= 50)
  +15  utente con segnali audio moderati (risk_score audio 25-49)
  +20  anomalia geografica GPS > 800 km
  +10  anomalia geografica GPS 400-800 km

Soglia minima per passare all'agente: score >= 20.
"""

from __future__ import annotations
import re
import os
import json
import pandas as pd
from math import radians, cos, sin, asin, sqrt


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


PHISHING_PATTERNS = [
    r"amaz0n", r"paypa1", r"verify.*account", r"suspicious.*login",
    r"account.*lock", r"urgent", r"verify.*identity", r"prevent.*lock",
    r"secure.*account", r"suspicious.*sign.?in", r"click.*link",
    r"confirm.*password", r"reset.*password", r"unusual.*activity",
]

AUDIO_FRAUD_PATTERNS = [
    r"urgent", r"immediately", r"transfer", r"bank.*account", r"pin\b",
    r"otp", r"password", r"verify", r"suspend", r"block.*card",
    r"refund", r"prize", r"won",
    r"subito", r"urgente", r"bonifico", r"conto", r"sospeso",
    r"verifica", r"codice", r"clicca", r"premio", r"vinto",
    r"virement", r"compte", r"suspendu", r"verification",
]


def _phishing_score_by_user(sms_data, mail_data) -> dict[str, int]:
    scores: dict[str, int] = {}

    def _scan(items, field_keys):
        if isinstance(items, dict):
            for uid, msgs in items.items():
                for msg in msgs:
                    text = ""
                    for k in field_keys:
                        text = str(msg.get(k, "")).lower()
                        if text:
                            break
                    for p in PHISHING_PATTERNS:
                        if re.search(p, text):
                            scores[uid] = scores.get(uid, 0) + 1
                            break
        elif isinstance(items, list):
            for msg in items:
                uid = str(msg.get("user_id", "") or msg.get("recipient", ""))
                if not uid:
                    continue
                text = ""
                for k in field_keys:
                    text = str(msg.get(k, "")).lower()
                    if text:
                        break
                for p in PHISHING_PATTERNS:
                    if re.search(p, text):
                        scores[uid] = scores.get(uid, 0) + 1
                        break

    _scan(sms_data,  ["sms", "text", "content"])
    _scan(mail_data, ["mail", "body", "content"])
    return scores


def _geo_anomaly_by_user(users, locations) -> dict[str, float]:
    user_map: dict[str, dict] = {}
    for u in (users if isinstance(users, list) else []):
        uid = str(u.get("id", "") or u.get("user_id", ""))
        user_map[uid] = u
        if u.get("iban"):   user_map[u["iban"]]   = u
        if u.get("biotag"): user_map[u["biotag"]] = u

    max_dist: dict[str, float] = {}
    for loc in (locations if isinstance(locations, list) else []):
        user = (
            user_map.get(str(loc.get("biotag", "")))
            or user_map.get(str(loc.get("user_id", "")))
            or user_map.get(str(loc.get("iban", "")))
        )
        if not user:
            continue
        uid = str(user.get("id", "") or user.get("user_id", ""))
        home_lat, home_lng = user.get("lat"), user.get("lng")
        if not home_lat or not home_lng:
            continue
        dist = _haversine(home_lat, home_lng,
                          loc.get("lat", home_lat), loc.get("lng", home_lng))
        if dist > max_dist.get(uid, 0):
            max_dist[uid] = dist
    return max_dist


def _audio_risk_by_user(audio_files: list, users: list) -> dict[str, int]:
    """Trascrive gli audio e calcola un risk_score per nome utente.
    Mappa poi nome -> user_id tramite il dataset users.

    Restituisce {user_id: audio_risk_score} dove audio_risk_score e'
    il numero di pattern di vishing trovati (non cappato).
    """
    if not audio_files:
        return {}

    # Mappa nome_cognome -> user_id cercando match nel dataset users
    name_to_uid: dict[str, str] = {}
    for u in (users if isinstance(users, list) else []):
        full_name = (
            f"{u.get('name', '')} {u.get('surname', '')}".strip().lower()
            or f"{u.get('first_name', '')} {u.get('last_name', '')}".strip().lower()
        )
        if full_name:
            uid = str(u.get("id", "") or u.get("user_id", ""))
            name_to_uid[full_name] = uid

    try:
        from transcriber import transcribe
    except ImportError:
        print("[prescorer] faster-whisper non disponibile, skip audio.")
        return {}

    audio_scores: dict[str, int] = {}  # user_id -> count segnali

    for af in audio_files:
        user_name = af.get("user_name", "").lower()  # es. 'guido dohn'
        uid = name_to_uid.get(user_name, "")
        if not uid:
            # fallback: cerca parziale
            for name, u in name_to_uid.items():
                if user_name in name or name in user_name:
                    uid = u
                    break

        transcript = transcribe(af["path"])
        text_lower = transcript.lower()
        count = sum(1 for p in AUDIO_FRAUD_PATTERNS if re.search(p, text_lower))

        if uid and count > 0:
            audio_scores[uid] = audio_scores.get(uid, 0) + count

    return audio_scores


def compute_risk_scores(data: dict, z_threshold: float = 2.2) -> pd.DataFrame:
    """Calcola risk_score per ogni transazione includendo segnale audio.
    Restituisce DataFrame ordinato per risk_score decrescente, solo score >= 20.
    """
    txs: pd.DataFrame = data["transactions"]
    if not isinstance(txs, pd.DataFrame):
        txs = pd.DataFrame(txs)
    txs = txs.copy()

    txs["timestamp"]     = pd.to_datetime(txs["timestamp"], errors="coerce")
    txs["hour"]          = txs["timestamp"].dt.hour
    txs["amount"]        = pd.to_numeric(txs["amount"], errors="coerce")
    txs["balance_after"] = pd.to_numeric(txs["balance_after"], errors="coerce")

    # z-score per utente
    user_stats = txs.groupby("sender_id")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["sender_id", "mean_amount", "std_amount"]
    txs = txs.merge(user_stats, on="sender_id", how="left")
    txs["z_score"] = (txs["amount"] - txs["mean_amount"]) / txs["std_amount"].replace(0, 1)

    # frequenza tipo transazione
    type_counts = txs.groupby(["sender_id", "transaction_type"]).size().reset_index(name="cnt")
    user_totals = txs.groupby("sender_id").size().reset_index(name="tot")
    type_freq   = type_counts.merge(user_totals, on="sender_id")
    type_freq["freq"] = type_freq["cnt"] / type_freq["tot"].replace(0, 1)
    txs = txs.merge(
        type_freq[["sender_id", "transaction_type", "freq"]],
        on=["sender_id", "transaction_type"], how="left",
    )

    phishing_scores = _phishing_score_by_user(
        data.get("sms", []), data.get("mails", [])
    )
    geo_dist = _geo_anomaly_by_user(
        data.get("users", []), data.get("locations", [])
    )
    # Audio: solo se ci sono file
    audio_files = data.get("audio_files", [])
    print(f"[prescorer] Trascrizione di {len(audio_files)} file audio...")
    audio_risk = _audio_risk_by_user(audio_files, data.get("users", []))
    print(f"[prescorer] Utenti con segnali audio: {len(audio_risk)}")

    rows = []
    for _, row in txs.iterrows():
        score = 0
        reasons = []
        uid = str(row.get("sender_id", ""))

        if pd.notna(row["z_score"]) and row["z_score"] > z_threshold:
            score += 30
            reasons.append(f"importo anomalo (z={round(row['z_score'], 2)})")

        if pd.notna(row["hour"]) and 0 <= int(row["hour"]) < 6:
            score += 20
            reasons.append(f"orario notturno ({int(row['hour'])}:xx)")

        if pd.notna(row["balance_after"]) and row["balance_after"] < 50:
            score += 25
            reasons.append(f"saldo critico ({row['balance_after']})")

        freq = row.get("freq")
        if pd.notna(freq) and freq < 0.1 and score > 0:
            score += 15
            reasons.append("tipo raro per utente")

        ph = phishing_scores.get(uid, 0)
        if ph >= 2:
            score += 20
            reasons.append(f"{ph} messaggi phishing (SMS/mail)")
        elif ph == 1:
            score += 10
            reasons.append("1 messaggio phishing")

        # segnale audio (vishing)
        ar = audio_risk.get(uid, 0)
        if ar >= 2:
            score += 25
            reasons.append(f"vishing audio: {ar} pattern rilevati")
        elif ar == 1:
            score += 15
            reasons.append("vishing audio: 1 pattern rilevato")

        dist = geo_dist.get(uid, 0)
        if dist > 800:
            score += 20
            reasons.append(f"GPS lontano {round(dist)} km")
        elif dist > 400:
            score += 10
            reasons.append(f"GPS a {round(dist)} km")

        rows.append({
            "transaction_id":   row.get("transaction_id", ""),
            "sender_id":        uid,
            "amount":           row.get("amount", ""),
            "hour":             int(row["hour"]) if pd.notna(row.get("hour")) else None,
            "balance_after":    row.get("balance_after", ""),
            "transaction_type": row.get("transaction_type", ""),
            "z_score":          round(row["z_score"], 2) if pd.notna(row.get("z_score")) else None,
            "risk_score":       min(score, 100),
            "risk_reasons":     reasons,
        })

    result = pd.DataFrame(rows)
    result = result[result["risk_score"] >= 20].sort_values("risk_score", ascending=False)
    return result
