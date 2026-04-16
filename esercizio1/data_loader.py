import pandas as pd
import json
from config import TRANSACTIONS_FILE, LOCATIONS_FILE, MAILS_FILE, SMS_FILE, USERS_FILE


def load_transactions() -> pd.DataFrame:
    """Carica le transazioni dal CSV."""
    df = pd.read_csv(TRANSACTIONS_FILE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_json(path: str) -> list | dict:
    """Carica un file JSON generico."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all() -> dict:
    """Carica tutti i dataset e li restituisce come dizionario."""
    transactions = load_transactions()
    locations = load_json(LOCATIONS_FILE)
    mails = load_json(MAILS_FILE)
    sms = load_json(SMS_FILE)
    users = load_json(USERS_FILE)

    print(f"[loader] Transazioni caricate: {len(transactions)}")
    print(f"[loader] Utenti: {len(users) if isinstance(users, list) else 'dict'}")
    print(f"[loader] Location records: {len(locations) if isinstance(locations, list) else 'dict'}")

    return {
        "transactions": transactions,
        "locations": locations,
        "mails": mails,
        "sms": sms,
        "users": users,
    }
