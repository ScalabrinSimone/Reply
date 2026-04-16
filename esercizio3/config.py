import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter config
OPENROUTER_API_KEY  = os.getenv("OpenRouterKey")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# PREMIUM MODEL: claude-opus-4-5 (seconda API key)
MODEL_ID = "anthropic/claude-opus-4-5"

# Langfuse config
LANGFUSE_PUBLIC_KEY = os.getenv("langfuse_publicKey")
LANGFUSE_SECRET_KEY = os.getenv("langfuse_privateKey")
LANGFUSE_HOST       = os.getenv("langfuse_host")

# Paths (relativi alla cartella esercizio3/)
DATA_DIR             = "data"
TRANSACTIONS_FILE    = f"{DATA_DIR}/transactions-5-5.csv"
# Dataset location piu' ricco: combinazione di locations-2-2.json + locations-2.json (se presente)
LOCATIONS_FILE       = f"{DATA_DIR}/locations-2-2.json"
EXTRA_LOCATIONS_FILE = f"{DATA_DIR}/locations-2.json"
MAILS_FILE           = f"{DATA_DIR}/mails-3-3.json"
SMS_FILE             = f"{DATA_DIR}/sms-4-4.json"
USERS_FILE           = f"{DATA_DIR}/users-6-6.json"
AUDIO_DIR            = f"{DATA_DIR}/audio"
OUTPUT_FILE          = "output.txt"

# Modello Whisper locale (faster-whisper).
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
