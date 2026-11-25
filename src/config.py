from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEARCH_QUERY = 'ti:"traffic signal control" AND "reinforcement learning"'
MAX_PAPERS = 10
RAW_DIR = ROOT / "data/raw"
METADATA_PATH = ROOT / "data/metadata.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
DB_PATH = ROOT / "data/chroma_db"
COLLECTION_NAME = "research_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"
TOP_K = 5
