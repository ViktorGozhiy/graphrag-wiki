# ABOUTME: Hybrid retrieval — fuses deep vector and BM25 candidate pools with Reciprocal Rank Fusion.
# ABOUTME: Returns fused chunk records tagged with which retrievers found each one.

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.fusion import reciprocal_rank_fusion
from graphrag_wiki.keyword import BM25Index
from graphrag_wiki.vector import search_vectors


def search_hybrid(query, k=5, pool=30, weights=None, bm25=None):
    """Return the top-k chunk records fused from vector and BM25 retrieval.

    Draws `pool` candidates from each retriever, fuses by RRF, and returns the
    top-k records each carrying its fused `score` and a `found_by` tag.
    """
    bm25 = bm25 or BM25Index(iter_chunks(load_corpus(CORPUS_PATH)))
    vector_hits = search_vectors(query, k=pool)
    bm25_hits = bm25.search(query, k=pool)

    records = {hit["chunk_id"]: hit for hit in bm25_hits}
    records.update({hit["chunk_id"]: hit for hit in vector_hits})
    vector_ids = {hit["chunk_id"] for hit in vector_hits}
    bm25_ids = {hit["chunk_id"] for hit in bm25_hits}

    fused = reciprocal_rank_fusion(
        [[hit["chunk_id"] for hit in vector_hits], [hit["chunk_id"] for hit in bm25_hits]],
        weights=weights,
    )

    results = []
    for chunk_id, score in fused[:k]:
        record = dict(records[chunk_id])
        record["score"] = score
        in_vector, in_bm25 = chunk_id in vector_ids, chunk_id in bm25_ids
        record["found_by"] = "V+B" if in_vector and in_bm25 else ("V" if in_vector else "B")
        results.append(record)
    return results
