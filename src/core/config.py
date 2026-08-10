import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=True)

# Base Paths (Updated for src/core structure)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "embeddings"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
CHUNK_SIZE = 350  # Number of words per chunk (optimal for 384-512 token embedding windows)
CHUNK_OVERLAP = 50 # Overlap to maintain context

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"


# Vector Store Paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# OCR Configuration
OCR_LANGUAGES = ["en"]

