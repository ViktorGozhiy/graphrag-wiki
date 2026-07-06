# ABOUTME: Reads the article corpus and turns it into the shared chunk records.
# ABOUTME: Each chunk carries a stable {pageid}:{index} id used across all retrievers.

import json

from graphrag_wiki.chunking import chunk_text


def load_corpus(path):
    """Load the JSONL corpus into a list of article records."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def iter_chunks(rows, chunk_size=1600, overlap=240):
    """Yield chunk records with a stable id and the source article's metadata."""
    for row in rows:
        for index, text in enumerate(chunk_text(row["text"], chunk_size, overlap)):
            yield {
                "chunk_id": f"{row['pageid']}:{index}",
                "pageid": row["pageid"],
                "title": row["title"],
                "url": row["url"],
                "chunk_index": index,
                "text": text,
            }
