# ABOUTME: Hybrid retrieval — fuses deep vector and BM25 candidate pools with Reciprocal Rank Fusion.
# ABOUTME: Prints the fused top-k and tags which retrievers found each chunk (V, B, or both).

import sys

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.fusion import reciprocal_rank_fusion
from graphrag_wiki.keyword import BM25Index
from graphrag_wiki.vector import search_vectors

POOL = 30  # candidates drawn from each retriever before fusion
TOP_K = 5  # results shown after fusion


def main():
    query = " ".join(sys.argv[1:]).strip() or "the denarius and Roman coinage"

    vector_hits = search_vectors(query, k=POOL)
    bm25_hits = BM25Index(iter_chunks(load_corpus(CORPUS_PATH))).search(query, k=POOL)

    records = {hit["chunk_id"]: hit for hit in bm25_hits}
    records.update({hit["chunk_id"]: hit for hit in vector_hits})
    vector_ids = {hit["chunk_id"] for hit in vector_hits}
    bm25_ids = {hit["chunk_id"] for hit in bm25_hits}

    fused = reciprocal_rank_fusion([
        [hit["chunk_id"] for hit in vector_hits],
        [hit["chunk_id"] for hit in bm25_hits],
    ])

    print(f"Q: {query}   (pool={POOL} each, top {TOP_K})\n")
    for chunk_id, score in fused[:TOP_K]:
        record = records[chunk_id]
        if chunk_id in vector_ids and chunk_id in bm25_ids:
            tag = "V+B"
        elif chunk_id in vector_ids:
            tag = "V"
        else:
            tag = "B"
        snippet = " ".join(record["text"].split())[:180]
        print(f"[{score:.4f}] [{tag:>3}] {record['title']}  ({chunk_id})")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
