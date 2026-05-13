from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
APP_ENV     = os.getenv("APP_ENV")

MODEL = "llama3.1"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
N_RETRIEVAL_RESULTS = 3