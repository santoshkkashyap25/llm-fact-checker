# config.py
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACTS_CSV_PATH = DATA_DIR / "trusted_facts.csv"
VECTOR_INDEX_PATH = DATA_DIR / "faiss_index.bin"

EMBEDDING_MODEL = 'all-mpnet-base-v2'

LLM_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# --- RAG Pipeline Parameters ---
TOP_K_RESULTS = 3  # Number of similar facts to retrieve
CONFIDENCE_THRESHOLD = 0.55 # Similarity score threshold to consider a fact relevant

APP_TITLE = "LLM Powered Fact Checker"