# config.py
from pathlib import Path
from typing import Optional
import os

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACTS_CSV_PATH = DATA_DIR / "trusted_facts.csv"
VECTOR_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METRICS_PATH = BASE_DIR / "metrics.jsonl"
CACHE_PATH = DATA_DIR / "query_cache.json"

# --- Models ---
# Faster, smaller
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 384d, very fast

# Better accuracy, larger  
# EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # 768d

# Multilingual support
# EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# For faster responses (less accurate)
# LLM_REPO_ID = "google/flan-t5-large"

# For better accuracy (slower)
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# For balanced performance
# LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
SPACY_MODEL = "en_core_web_md"

# --- RAG Pipeline Parameters ---
TOP_K_RESULTS = 5  
CONFIDENCE_THRESHOLD = 0.50
SIMILARITY_MATCH_THRESHOLD = 0.85

# --- Cache Settings ---
CACHE_ENABLED = True
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 3600  # 1 hour

# --- App Settings ---
APP_TITLE = "LLM-Powered Fact Checker"
APP_VERSION = "2.0.0"
MAX_INPUT_LENGTH = 1000

# --- Data Scraping Settings ---
PIB_RSS_URL = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1"
SCRAPE_ENABLED = os.getenv("ENABLE_SCRAPING", "false").lower() == "true"
SCRAPE_INTERVAL_HOURS = 24
