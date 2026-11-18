"""Configuration for SpecRAG client and data preparation."""

import os

# OpenAI-compatible API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "EMPTY"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_EMBEDDING_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL", "http://localhost:8001/v1")
OPENAI_CHAT_BASE_URL = os.getenv("OPENAI_CHAT_BASE_URL", OPENAI_BASE_URL)

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "500"))
EMBEDDING_NUM_WORKERS = int(os.getenv("EMBEDDING_NUM_WORKERS", "4"))

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
