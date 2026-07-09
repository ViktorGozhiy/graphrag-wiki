# ABOUTME: Batch-extracts the knowledge graph from every corpus chunk into a JSONL file.
# ABOUTME: Resumable (skips chunks already written) and concurrent, so the full corpus finishes fast.

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.graph_extraction import extract

OUTPUT_PATH = "data/graph/extractions.jsonl"

# Extraction is network-bound, so many chunks run against the API at once.
MAX_WORKERS = 12

# Bail out when the endpoint is clearly dead (e.g. quota exhausted): many failures
# and nothing succeeding, so there is no point working through the rest.
FAILURE_ABORT = 20


def done_chunk_ids(path):
    """Chunk ids already extracted, so a resumed run skips them."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["chunk_id"] for line in handle if line.strip()}


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    chunks = list(iter_chunks(load_corpus(CORPUS_PATH)))
    done = done_chunk_ids(OUTPUT_PATH)
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in done]
    print(f"chunks: {len(chunks)}  done: {len(done)}  pending: {len(pending)}", flush=True)

    successes = 0
    failures = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out, ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(extract, chunk["chunk_id"], chunk["text"]): chunk for chunk in pending}
        for index, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            chunk_id = chunk["chunk_id"]
            try:
                result = future.result()
            except Exception as error:
                failures += 1
                print(
                    f"[{index}/{len(pending)}] {chunk_id} FAILED "
                    f"({type(error).__name__}: {error}) — will retry on next run",
                    flush=True,
                )
                if successes == 0 and failures >= FAILURE_ABORT:
                    print(
                        f"{failures} failures and nothing extracted — endpoint down; stopping",
                        flush=True,
                    )
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                continue
            successes += 1
            out.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "title": chunk["title"],
                        "entities": result["entities"],
                        "relations": result["relations"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            print(
                f"[{index}/{len(pending)}] {chunk_id} ({chunk['title']}): "
                f"{len(result['entities'])} ent, {len(result['relations'])} rel",
                flush=True,
            )

    print(f"done — {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
