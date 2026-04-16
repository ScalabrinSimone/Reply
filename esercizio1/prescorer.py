"""FIX 2: Pre-scorer deterministico in Python.

Calcola un risk_score numerico per ogni transazione PRIMA di passare
all'agente. In questo modo:
- L'agente riceve gia' una lista ridotta e ordinata per rischio.
- Non deve fare calcoli statistici da solo: si concentra solo sul
  giudizio finale (includere / escludere) usando il contesto qualitativo.
- Il contesto del modello rimane piccolo e preciso.

Risk score per transazione (0-100, somma capped):
  +30  importo anomalo (z-score > 2.2)
  +20  orario notturno (00:00-05:59)
  +25  saldo residuo critico (< 50)
  +15  tipo di transazione raro per utente (freq < 10%, solo se altro segnale)
  +20  utente con phishing in SMS/mail (calcolato separatamente e iniettato)
  +20  anomalia geografica GPS (calcolata separatamente e iniettata)

Soglia minima per passare all'agente: score >= 20 (almeno un segnale).
"""

from __future__ import annotations
import re
import json
import pandas as pd
from math import radians, cos, sin, asin, sqrt


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Phishing score per utente (dai SMS e mail)
# ---------------------------------------------------------------------------
PHISHING_PATTERNS = [
    r"amaz0n", r"paypa1", r"verify.*account", r"suspicious.*login",
    r"account.*lock", r"urgent", r"verify.*identity", r"prevent.*lock",
    r"secure.*account", r"suspicious.*sign.?in", r"click.*link",
    r"confirm.*password", r"reset.*password", r"unusual.*activity",
]


def _phishing_score_by_user(sms_data, mail_data) -> dict[str, int]:
    """Restituisce {user_id: phishing_count} contando messaggi sospetti."""
    scores: dict[str, int] = {}

    def _scan(items, field_keys):
        if isinstance(items, dict):
            # {user_id: [msgs]}
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


# ---------------------------------------------------------------------------
# Geo anomaly: per ogni utente calcola distanza massima casa-GPS nelle 24h
# ---------------------------------------------------------------------------
def _geo_anomaly_by_user(users, locations) -> dict[str, float]:
    """Restituisce {user_id: max_dist_km} distanza massima rilevata."""
    # Mappa utente -> info
    user_map: dict[str, dict] = {}
    for u in (users if isinstance(users, list) else []):
        uid = str(u.get("id", "") or u.get("user_id", ""))
        user_map[uid] = u
        # anche per iban e biotag
        if u.get("iban"):
            user_map[u["iban"]] = u
        if u.get("biotag"):
            user_map[u["biotag"]] = u

    max_dist: dict[str, float] = {}
    for loc in (locations if isinstance(locations, list) else []):
        # prova a trovare l'utente tramite biotag, user_id, iban
        user = (
            user_map.get(str(loc.get("biotag", "")))
            or user_map.get(str(loc.get("user_id", "")))
            or user_map.get(str(loc.get("iban", "")))
        )
        if not user:
            continue
        uid = str(user.get("id", "") or user.get("user_id", ""))
        home_lat = user.get("lat")
        home_lng = user.get("lng")
        if not home_lat or not home_lng:
            continue
        dist = _haversine(home_lat, home_lng,
                          loc.get("lat", home_lat), loc.get("lng", home_lng))
        if dist > max_dist.get(uid, 0):
            max_dist[uid] = dist
    return max_dist


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def compute_risk_scores(data: dict, z_threshold: float = 2.2) -> pd.DataFrame:
    """Calcola un risk_score per ogni transazione e restituisce
    un DataFrame ordinato per rischio decrescente.

    Colonne extra aggiunte:
      risk_score   int  0-100+
      risk_reasons list  lista di stringhe descrittive
    """
    txs: pd.DataFrame = data["transactions"]
    if not isinstance(txs, pd.DataFrame):
        txs = pd.DataFrame(txs)
    txs = txs.copy()

    txs["timestamp"]    = pd.to_datetime(txs["timestamp"], errors="coerce")
    txs["hour"]         = txs["timestamp"].dt.hour
    txs["amount"]       = pd.to_numeric(txs["amount"], errors="coerce")
    txs["balance_after"] = pd.to_numeric(txs["balance_after"], errors="coerce")

    # --- z-score per utente ---
    user_stats = txs.groupby("sender_id")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["sender_id", "mean_amount", "std_amount"]
    txs = txs.merge(user_stats, on="sender_id", how="left")
    txs["z_score"] = (txs["amount"] - txs["mean_amount"]) / txs["std_amount"].replace(0, 1)

    # --- frequenza tipo transazione per utente ---
    type_counts = txs.groupby(["sender_id", "transaction_type"]).size().reset_index(name="cnt")
    user_totals = txs.groupby("sender_id").size().reset_index(name="tot")
    type_freq   = type_counts.merge(user_totals, on="sender_id")
    type_freq["freq"] = type_freq["cnt"] / type_freq["tot"].replace(0, 1)
    txs = txs.merge(
        type_freq[["sender_id", "transaction_type", "freq"]],
        on=["sender_id", "transaction_type"], how="left",
    )

    # --- phishing e geo pre-calcolati ---
    phishing_scores = _phishing_score_by_user(
        data.get("sms", []), data.get("mails", [])
    )
    geo_dist = _geo_anomaly_by_user(
        data.get("users", []), data.get("locations", [])
    )

    # --- calcolo score riga per riga ---
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

        # bonus phishing comunicazioni
        ph = phishing_scores.get(uid, 0)
        if ph >= 2:
            score += 20
            reasons.append(f"utente con {ph} messaggi phishing")
        elif ph == 1:
            score += 10
            reasons.append("utente con 1 messaggio phishing")

        # bonus geo anomaly
        dist = geo_dist.get(uid, 0)
        if dist > 800:
            score += 20
            reasons.append(f"GPS lontano {round(dist)} km dalla residenza")
        elif dist > 400:
            score += 10
            reasons.append(f"GPS a {round(dist)} km dalla residenza")

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
