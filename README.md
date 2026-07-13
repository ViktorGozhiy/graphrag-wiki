# graphrag-wiki

A hands-on, step-by-step build of a Retrieval-Augmented Generation (RAG) system over a small
Wikipedia corpus, working toward GraphRAG. It is a learning project: each retrieval technique is
implemented transparently so the mechanics are easy to read and compare.

**Current state:** a working GraphRAG system. The baseline is hybrid retrieval (dense vectors +
BM25 fused with Reciprocal Rank Fusion) feeding a locally-run LLM that answers with citations. On
top of it, a knowledge graph — typed entities and relations extracted from every chunk, loaded
into Neo4j, with duplicate entities resolved — powers graph-aware retrieval that surfaces bridging
passages a flat search ranks too low to find. `scripts/ask_graph.py` answers a question both ways,
side by side, so the graph's contribution is visible.

## How it works

**Indexing** — build the retrievers once from the corpus:

```
Wikipedia ─download─▶ corpus.jsonl ─chunk─▶ chunks (shared unit: {pageid}:{index})
                                              ├─ embed (bge-small) ─▶ Qdrant (vectors)
                                              ├─ BM25 index (in-memory)
                                              └─ LLM extraction ─▶ Neo4j graph
                                                 (typed entities + relations;
                                                  SAME_AS links resolve duplicates)
```

**Answering** — a question runs through both retrieval paths, fused into one cited answer:

```
question
   ├─ hybrid:  vector + BM25 ─▶ RRF ─▶ top-5 passages ───────────────┐
   └─ graph:   LLM names the question's entities ─▶ link to nodes     │
              ─▶ traverse their SAME_AS-expanded neighborhood         │
              ─▶ candidate chunks ─▶ rerank by relevance              │
              ─▶ passages from articles hybrid missed ────────────────┤
                                                                      ▼
                                                 LLM (Ollama) ─▶ cited answer
```

Every retriever operates on the **same chunk unit**, keyed by a stable `{pageid}:{chunk_index}` id,
so results can be compared fairly and fused. The graph earns its place by reaching bridging passages
in lexically-distant articles that vector and keyword search rank too low to surface.

## Tech stack

- [uv](https://docs.astral.sh/uv/) — environment and dependency management (Python 3.12)
- [Qdrant](https://qdrant.tech/) — vector store
- [Neo4j](https://neo4j.com/) — graph store for the extracted knowledge graph
- [Ollama](https://ollama.com/) — local LLM serving for RAG answers (`gemma3:4b` by default)
- [OpenAI API](https://platform.openai.com/) — `gpt-4.1-mini` for graph extraction, entity
  resolution, and query entity linking (Structured Outputs)
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

Extract typed entities and relations from every chunk into `data/graph/extractions.jsonl`. This runs
on the OpenAI API (`gpt-4.1-mini`) using Structured Outputs, so both passes return schema-valid JSON.
Put your key in a `.env` at the repo root (it is git-ignored):

```bash
echo "OPENAI_API_KEY=sk-..." > .env
bash scripts/run_extraction.sh     # extracts the whole corpus, resuming after any stop
```

Each chunk goes through two schema-constrained LLM passes — entities first, then relations between
those entities — validated against a fixed vocabulary of node and relation types. Chunks are
extracted concurrently, so the full corpus finishes in about an hour. The run is **resumable**: it
skips chunks already written, so re-running continues where it left off and picks up any that failed.
`run_extraction.sh` wraps `extract_graph.py` (one pass) in a resume loop; use the inner script
directly if you prefer to drive it yourself.

## Building the graph

With Neo4j running and `data/graph/extractions.jsonl` in place (both included), load the graph and
resolve duplicate entities:

```bash
# Load entities and relations into Neo4j (idempotent MERGE — safe to re-run)
uv run python scripts/load_graph.py

# Resolve duplicate entities: block candidate pairs, let the LLM decide, write SAME_AS links
set -a && source .env && set +a          # entity resolution calls the OpenAI API
uv run python scripts/resolve_entities.py
```

Extraction names the same entity many ways ("Caracalla", "Emperor Caracalla"). Resolution finds
likely-duplicate pairs cheaply — by shared name tokens and by embedding similarity, both filtered
to hub entities — then asks the LLM to decide which truly denote the same thing. Confirmed pairs get
a reversible `SAME_AS` link carrying a canonical name, rather than being destructively merged. The
run is **resumable** via `data/graph/resolutions.jsonl`, and links are re-derived on a fresh graph.

## Graph-aware retrieval

```bash
set -a && source .env && set +a          # query entity linking calls the OpenAI API
uv run python scripts/ask_graph.py "Which emperor granted Roman citizenship to all free \
  inhabitants of the empire, and how did that grant affect Roman taxation?"
```

This answers the question **twice** — with baseline hybrid retrieval and with graph-aware retrieval
— and prints both, so the difference is visible. The graph path asks the LLM which entities the
question is about, links them to graph nodes, walks their `SAME_AS`-expanded neighborhood to gather
connected passages, reranks those by relevance, and adds the ones from articles hybrid missed as a
separate context block. On the sample question, baseline answers the first hop (Caracalla granted
citizenship) but not the second; graph-aware retrieval reaches the lexically-distant *Taxation in
ancient Rome* passage and answers that the grant widened the tax base.

## Project layout

```
graphrag_wiki/          # reusable pipeline modules
  chunking.py           #   recursive text splitter (the shared chunk unit)
  wikitext.py           #   clean Wikipedia extracts into narrative text
  corpus.py             #   load corpus, produce chunk records
  vector.py             #   dense search over Qdrant; rerank a candidate set by relevance
  keyword.py            #   in-memory BM25 index
  fusion.py             #   Reciprocal Rank Fusion
  hybrid.py             #   fused vector + BM25 retrieval
  display.py            #   shared terminal formatting for retrieval results
  llm.py                #   shared OpenAI structured-output call (throttled, retried)
  graph_schema.py       #   node and relation types for extraction
  graph_extraction.py   #   two-pass entity/relation extraction via the LLM
  graph_store.py        #   load entities and relations into Neo4j; SAME_AS links
  entity_resolution.py  #   find and resolve duplicate entities (blocking + LLM decision)
  graph_retrieval.py    #   graph-aware retrieval: link query, traverse, gather candidates
  generation.py         #   grounded answer generation with the local model
  config.py             #   paths, model names, endpoints
scripts/                # runnable entry points (download, ingest, search, ask, ask_graph,
                        #   extract_graph, run_extraction, load_graph, resolve_entities)
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
