# ABOUTME: Dense vector search over the Qdrant chunk collection.
# ABOUTME: Embeds the query with the retrieval prefix and returns chunk records with scores.

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
