# ABOUTME: Batch-extracts the knowledge graph from every corpus chunk into a JSONL file.
# ABOUTME: Resumable — skips chunks already written — so a long CPU run survives interruptions.

import json
import os

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.graph_extraction import extract

OUTPUT_PATH = "data/graph/extractions.jsonl"


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

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for index, chunk in enumerate(pending, 1):
            chunk_id = chunk["chunk_id"]
            try:
                result = extract(chunk_id, chunk["text"])
            except Exception as error:
                print(
                    f"[{index}/{len(pending)}] {chunk_id} FAILED "
                    f"({type(error).__name__}) — will retry on next run",
                    flush=True,
                )
                continue
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
