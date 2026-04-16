"""Pre-scorer deterministico v2.

Miglioramenti rispetto a v1:
  - z_threshold abbassato a 1.8 (meno falsi negativi)
  - NUOVO: segnale 'IBAN destinatario mai visto dall'utente' (+20)
  - NUOVO: spike di frequenza (>=3 tx nello stesso giorno) (+15)
  - NUOVO: importo alto assoluto (>= 3000) sempre incluso come candidato (+10)
    per non perdere frodi ad alto impatto economico con z-score basso
  - geo soglia abbassata: >300 km (+10), >600 km (+20)
  - audio: soglia per bonus alto abbassata a >=2 pattern

Risk score per transazione (somma, capped 100):
  +30  z-score importo > 1.8
  +20  orario notturno 00-05
  +25  saldo residuo < 50
  +15  tipo raro per utente (freq < 10%, solo se score > 0)
  +20  >= 2 messaggi phishing
  +10  1 messaggio phishing
  +25  vishing audio >= 2 pattern
  +15  vishing audio 1 pattern
  +20  GPS > 600 km dalla residenza
  +10  GPS 300-600 km
  +20  IBAN destinatario nuovo per l'utente (mai visto prima)
  +15  spike frequenza (>= 3 tx stesso giorno)
  +10  importo alto assoluto (>= 3000, floor di inclusione)

Soglia minima per passare all'agente: score >= 10
(abbassata per catturare frodi ad alto importo con pochi segnali).
"""

from __future__ import annotations
import re
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
    r"bank.*detail", r"wire.*transfer", r"gift.*card", r"invoice",
    r"truffa", r"frode", r"bloccato", r"sospeso", r"verifica.*identit",
]

AUDIO_FRAUD_PATTERNS = [
    r"urgent", r"immediately", r"transfer", r"bank.*account", r"pin\b",
    r"otp", r"password", r"verify", r"suspend", r"block.*card",
    r"refund", r"prize", r"won",
    r"subito", r"urgente", r"bonifico", r"conto", r"sospeso",
    r"verifica", r"codice", r"clicca", r"premio", r"vinto",
    r"virement", r"compte", r"suspendu", r"verification",
    r"compromis", r"fraude", r"code.*secret",
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
    """Trascrive gli audio con faster-whisper e conta pattern di vishing.
    Restituisce {user_id: n_pattern}.
    """
    if not audio_files:
        return {}

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

    audio_scores: dict[str, int] = {}
    for af in audio_files:
        user_name = af.get("user_name", "").lower()
        uid = name_to_uid.get(user_name, "")
        if not uid:
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


def _new_iban_by_user(txs: pd.DataFrame) -> dict[str, set]:
    """Per ogni utente costruisce l'insieme degli IBAN destinatari 'abituali'
    (visti in almeno 2 transazioni precedenti).
    Restituisce {user_id: set_of_new_ibans} ovvero IBAN visti per la prima volta.
    """
    if "receiver_id" not in txs.columns:
        return {}

    # Conta quante volte ogni utente ha inviato a ciascun IBAN destinatario
    counts = (
        txs.groupby(["sender_id", "receiver_id"])
        .size()
        .reset_index(name="n")
    )
    # IBAN "abituali": n >= 2
    habitual = set(
        zip(counts[counts["n"] >= 2]["sender_id"],
            counts[counts["n"] >= 2]["receiver_id"])
    )
    # Tutti gli IBAN usati
    all_pairs = set(zip(txs["sender_id"], txs["receiver_id"]))
    # Nuovi = usati ma non abituali
    new_pairs = all_pairs - habitual
    result: dict[str, set] = {}
    for s, r in new_pairs:
        result.setdefault(str(s), set()).add(str(r))
    return result


def _frequency_spike_by_user(txs: pd.DataFrame) -> set[str]:
    """Restituisce l'insieme dei sender_id con >= 3 transazioni nello stesso giorno.
    Indica comportamento insolito (es. account compromesso che svuota il saldo).
    """
    if "timestamp" not in txs.columns:
        return set()
    txs = txs.copy()
    txs["date"] = pd.to_datetime(txs["timestamp"], errors="coerce").dt.date
    daily = txs.groupby(["sender_id", "date"]).size().reset_index(name="n")
    spiked = daily[daily["n"] >= 3]["sender_id"].unique()
    return set(str(s) for s in spiked)


def compute_risk_scores(data: dict, z_threshold: float = 1.8) -> pd.DataFrame:
    """Calcola risk_score per ogni transazione (v2: più segnali, soglia più bassa).
    Restituisce DataFrame ordinato per risk_score decrescente, solo score >= 10.
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

    # segnali pre-calcolati
    phishing_scores = _phishing_score_by_user(
        data.get("sms", []), data.get("mails", [])
    )
    geo_dist      = _geo_anomaly_by_user(data.get("users", []), data.get("locations", []))
    new_ibans     = _new_iban_by_user(txs)
    spike_users   = _frequency_spike_by_user(txs)

    audio_files = data.get("audio_files", [])
    print(f"[prescorer] Trascrizione di {len(audio_files)} file audio...")
    audio_risk = _audio_risk_by_user(audio_files, data.get("users", []))
    print(f"[prescorer] Utenti con segnali audio: {len(audio_risk)}")

    rows = []
    for _, row in txs.iterrows():
        score = 0
        reasons = []
        uid = str(row.get("sender_id", ""))
        amt = row.get("amount", 0) or 0

        # --- segnali statistici ---
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

        # --- importo alto assoluto: floor per non perdere frodi economicamente gravi ---
        if pd.notna(amt) and amt >= 3000:
            score += 10
            reasons.append(f"importo alto assoluto ({round(amt, 2)})")

        # --- IBAN destinatario nuovo ---
        receiver = str(row.get("receiver_id", ""))
        if uid in new_ibans and receiver in new_ibans[uid]:
            score += 20
            reasons.append(f"IBAN destinatario mai visto prima ({receiver[:12]}...)")

        # --- spike di frequenza ---
        if uid in spike_users:
            score += 15
            reasons.append("spike frequenza (>=3 tx stesso giorno)")

        # --- phishing SMS/mail ---
        ph = phishing_scores.get(uid, 0)
        if ph >= 2:
            score += 20
            reasons.append(f"{ph} messaggi phishing (SMS/mail)")
        elif ph == 1:
            score += 10
            reasons.append("1 messaggio phishing")

        # --- vishing audio ---
        ar = audio_risk.get(uid, 0)
        if ar >= 2:
            score += 25
            reasons.append(f"vishing audio: {ar} pattern rilevati")
        elif ar == 1:
            score += 15
            reasons.append("vishing audio: 1 pattern rilevato")

        # --- geo anomaly (soglie abbassate) ---
        dist = geo_dist.get(uid, 0)
        if dist > 600:
            score += 20
            reasons.append(f"GPS lontano {round(dist)} km")
        elif dist > 300:
            score += 10
            reasons.append(f"GPS a {round(dist)} km")

        rows.append({
            "transaction_id":   row.get("transaction_id", ""),
            "sender_id":        uid,
            "amount":           amt,
            "hour":             int(row["hour"]) if pd.notna(row.get("hour")) else None,
            "balance_after":    row.get("balance_after", ""),
            "transaction_type": row.get("transaction_type", ""),
            "z_score":          round(row["z_score"], 2) if pd.notna(row.get("z_score")) else None,
            "risk_score":       min(score, 100),
            "risk_reasons":     reasons,
        })

    result = pd.DataFrame(rows)
    # Soglia abbassata a 10 per catturare frodi ad alto importo con pochi segnali
    result = result[result["risk_score"] >= 10].sort_values("risk_score", ascending=False)
    return result
