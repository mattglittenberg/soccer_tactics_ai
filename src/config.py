from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# --- Model Settings ---
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- RAG Settings ---
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
N_RETRIEVAL_RESULTS = 3

# --- Server Settings ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
APP_ENV = os.getenv("APP_ENV")