import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OpenRouterKey")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model: claude-3.5-haiku via OpenRouter
# Ottimo rapporto qualita'/costo per task di analisi strutturata.
# ~$0.001 per 1K token input, molto piu' economico di claude-3.5-sonnet.
MODEL_ID = "anthropic/claude-3-5-haiku"

# Langfuse config
LANGFUSE_PUBLIC_KEY = os.getenv("langfuse_publicKey")
LANGFUSE_SECRET_KEY = os.getenv("langfuse_privateKey")
LANGFUSE_HOST = os.getenv("langfuse_host")

# Paths (relativi alla cartella esercizio1/)
DATA_DIR = "data"
TRANSACTIONS_FILE = f"{DATA_DIR}/transactions-5.csv"
LOCATIONS_FILE = f"{DATA_DIR}/locations-2.json"
MAILS_FILE = f"{DATA_DIR}/mails-3.json"
SMS_FILE = f"{DATA_DIR}/sms-4.json"
USERS_FILE = f"{DATA_DIR}/users-6.json"
OUTPUT_FILE = "output.txt"
