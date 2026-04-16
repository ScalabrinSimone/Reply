import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OpenRouterKey")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# FIX 1: cambio modello.
# deepseek/deepseek-r1 ha chain-of-thought esplicito: ragiona passo passo
# prima di rispondere, ottimo per task di analisi multi-segnale.
# Costo molto basso su OpenRouter (~$0.0005/1K token input).
MODEL_ID = "deepseek/deepseek-r1"

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
