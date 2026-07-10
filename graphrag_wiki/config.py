# ABOUTME: Central configuration for corpus path, embedding model, and the vector store endpoint.
# ABOUTME: Shared by the ingest and search scripts (and the later keyword/graph pipelines).

from dotenv import load_dotenv

load_dotenv()  # read OPENAI_API_KEY (and any local overrides) from a .env in the repo root

CORPUS_PATH = "data/raw/corpus.jsonl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "wiki_chunks"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "graphragdev")  # local throwaway dev credentials, not a secret

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"

# Graph extraction runs on the OpenAI API (OPENAI_API_KEY from the environment / .env).
# gpt-4.1-mini balances cost with accurate typed extraction and supports the strict
# JSON-schema structured outputs the two-pass extractor depends on.
OPENAI_MODEL = "gpt-4.1-mini"
