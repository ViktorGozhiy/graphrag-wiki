# ABOUTME: Dense vector search over the Qdrant chunk collection.
# ABOUTME: Embeds the query with the retrieval prefix and returns chunk records with scores.

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from graphrag_wiki.config import EMBED_MODEL, QDRANT_COLLECTION, QDRANT_URL

_model = None


def _embedder():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL)
    return _model


def search_vectors(query, k=5, client=None):
    """Return the top-k chunk records for a query, each with a cosine score."""
    client = client or QdrantClient(url=QDRANT_URL)
    vector = next(iter(_embedder().query_embed([query])))
    hits = client.query_points(
        QDRANT_COLLECTION, query=vector.tolist(), limit=k, with_payload=True
    ).points
    return [{**hit.payload, "score": float(hit.score)} for hit in hits]


def rank_by_query(query, chunks):
    """Order chunk records by cosine similarity of their text to the query.

    Used to rerank an arbitrary candidate set (e.g. chunks reached through the graph)
    by relevance, embedding the query and each chunk with the same model.
    """
    if not chunks:
        return []
    model = _embedder()
    query_vector = next(iter(model.query_embed([query])))
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-9)

    scored = []
    for chunk, vector in zip(chunks, model.embed([chunk["text"] for chunk in chunks])):
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        scored.append((float(np.dot(vector, query_vector)), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored]
