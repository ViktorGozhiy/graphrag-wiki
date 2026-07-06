# ABOUTME: Hybrid retrieval — fuses deep vector and BM25 candidate pools with Reciprocal Rank Fusion.
# ABOUTME: Prints the fused top-k and tags which retrievers found each chunk (V, B, or both).

import sys

from graphrag_wiki.hybrid import search_hybrid

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip() or "the denarius and Roman coinage"
    hits = search_hybrid(query, k=TOP_K)

    print(f"Q: {query}   (top {TOP_K})\n")
    for hit in hits:
        snippet = " ".join(hit["text"].split())[:180]
        print(f"[{hit['score']:.4f}] [{hit['found_by']:>3}] {hit['title']}  ({hit['chunk_id']})")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
