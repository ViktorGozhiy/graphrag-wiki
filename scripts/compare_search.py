# ABOUTME: Runs one query through both vector and BM25 retrieval and shows them side by side.
# ABOUTME: Highlights where semantic and lexical search agree and where they diverge.

import sys

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.display import print_hits
from graphrag_wiki.keyword import BM25Index
from graphrag_wiki.vector import search_vectors

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip() or "the denarius and Roman coinage"

    vector_hits = search_vectors(query, k=TOP_K)
    bm25_hits = BM25Index(iter_chunks(load_corpus(CORPUS_PATH))).search(query, k=TOP_K)

    print(f"Q: {query}\n")
    print("── VECTOR (semantic) ──")
    print_hits(vector_hits)
    print("\n── BM25 (keyword) ──")
    print_hits(bm25_hits)

    vector_ids = {hit["chunk_id"] for hit in vector_hits}
    bm25_ids = {hit["chunk_id"] for hit in bm25_hits}
    overlap = sorted(vector_ids & bm25_ids)
    print(f"\noverlap in top-{TOP_K}: {len(overlap)} chunk(s): {overlap or '—'}")


if __name__ == "__main__":
    main()
