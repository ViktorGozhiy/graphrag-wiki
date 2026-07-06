# ABOUTME: Central configuration for corpus path, embedding model, and the vector store endpoint.
# ABOUTME: Shared by the ingest and search scripts (and the later keyword/graph pipelines).

CORPUS_PATH = "data/raw/corpus.jsonl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "wiki_chunks"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:12b"
