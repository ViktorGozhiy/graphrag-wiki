# graphrag-wiki

A hands-on, step-by-step build of a Retrieval-Augmented Generation (RAG) system over a small
Wikipedia corpus, working toward GraphRAG. It is a learning project: each retrieval technique is
implemented transparently so the mechanics are easy to read and compare.

**Current state:** a working baseline RAG — hybrid retrieval (dense vectors + BM25 fused with
Reciprocal Rank Fusion) feeding a locally-run LLM that answers with citations. Knowledge-graph
extraction (typed entities and relations per chunk) is implemented and runs through the same LLM.
**Planned next:** load the extracted graph into Neo4j and add graph-aware retrieval — the "Graph"
in GraphRAG. Neo4j already runs in the stack for that step.

## How it works

```
Wikipedia  ──download──▶  corpus.jsonl  ──chunk──▶  chunks (shared unit: {pageid}:{index})
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                                ▼                                ▼
                  embed (bge-small)                 BM25 index                    (planned) graph
                        │                                │
                  Qdrant (vectors)              in-memory keyword
                        └──────────────┬─────────────────┘
                                       ▼
                         Reciprocal Rank Fusion (hybrid)
                                       ▼
                        top-k context ──▶  LLM (Ollama) ──▶  cited answer
```

Every retriever operates on the **same chunk unit**, keyed by a stable `{pageid}:{chunk_index}` id,
so results can be compared fairly and fused.

## Tech stack

- [uv](https://docs.astral.sh/uv/) — environment and dependency management (Python 3.12)
- [Qdrant](https://qdrant.tech/) — vector store
- [Neo4j](https://neo4j.com/) — graph store (for the planned graph step)
- [Ollama](https://ollama.com/) — local LLM serving (`gemma3:4b` by default)
- [fastembed](https://github.com/qdrant/fastembed) with `BAAI/bge-small-en-v1.5` — embeddings
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — keyword retrieval

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
# 1. Install dependencies (creates the virtualenv)
uv sync

# 2. Start the infrastructure (Qdrant, Neo4j, Ollama)
docker compose up -d

# 3. Pull the local LLM
docker exec graphrag-ollama ollama pull gemma3:4b

# 4. Ingest the corpus into Qdrant (chunk + embed + upsert)
uv run python scripts/ingest_qdrant.py
```

The corpus (`data/raw/corpus.jsonl`, 40 articles about ancient Rome) is included, so step 4 works
straight away. To rebuild it from Wikipedia, run `uv run python scripts/download_corpus.py`.

## Usage

```bash
# Semantic (vector) search
uv run python scripts/search_qdrant.py "why did the western roman empire fall"

# Keyword (BM25) search
uv run python scripts/search_bm25.py "praetorian guard"

# Compare vector vs BM25 side by side
uv run python scripts/compare_search.py "money system of the empire"

# Hybrid retrieval (RRF over both)
uv run python scripts/search_hybrid.py "how emperors held power"

# Full RAG: retrieve context, ask the LLM, answer with citations
uv run python scripts/ask.py "why did the western roman empire fall"
```

> Inference runs on CPU by default. The first `ask.py` call loads the model into memory; expect
> tens of seconds per answer.

## Knowledge graph extraction

Extract typed entities and relations from every chunk into `data/graph/extractions.jsonl`:

```bash
uv run python scripts/extract_graph.py
```

Each chunk goes through two schema-constrained LLM passes — entities first, then relations between
those entities — validated against a fixed vocabulary of node and relation types. The run is
**resumable**: it skips chunks already written, so re-running continues where it left off and picks
up any that failed.

> This is heavy on CPU — roughly 1–2 minutes per chunk, so the full corpus (~1.7k chunks) takes many
> hours. Run it on an always-on machine (re-run after any interruption to finish), or extract a
> subset. Requires the Ollama stack with `gemma3:4b`.

## Project layout

```
graphrag_wiki/          # reusable pipeline modules
  chunking.py           #   recursive text splitter (the shared chunk unit)
  wikitext.py           #   clean Wikipedia extracts into narrative text
  corpus.py             #   load corpus, produce chunk records
  vector.py             #   dense search over Qdrant
  keyword.py            #   in-memory BM25 index
  fusion.py             #   Reciprocal Rank Fusion
  hybrid.py             #   fused vector + BM25 retrieval
  graph_schema.py       #   node and relation types for extraction
  graph_extraction.py   #   two-pass entity/relation extraction via the LLM
  config.py             #   paths, model names, endpoints
scripts/                # runnable entry points (download, ingest, search, ask, extract_graph)
tests/                  # unit tests for the pure logic
docker-compose.yml      # Qdrant + Neo4j + Ollama
```

## Tests

```bash
uv run pytest
```

## Corpus

The corpus is text from the English Wikipedia, available under
[CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Article titles and URLs are kept with
each chunk as source metadata.
