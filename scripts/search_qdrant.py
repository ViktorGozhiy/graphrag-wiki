# ABOUTME: Semantic search over the Qdrant chunk collection.
# ABOUTME: Embeds the query with the retrieval prefix and prints the top matches with their sources.

import sys

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from graphrag_wiki.config import EMBED_MODEL, QDRANT_COLLECTION, QDRANT_URL

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip() or "How did a Roman emperor gain and hold power?"

    model = TextEmbedding(model_name=EMBED_MODEL)
    vector = next(iter(model.query_embed([query])))

    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(
        QDRANT_COLLECTION, query=vector.tolist(), limit=TOP_K, with_payload=True
    ).points

    print(f"Q: {query}\n")
    for hit in hits:
        payload = hit.payload
        snippet = " ".join(payload["text"].split())[:220]
        print(f"[{hit.score:.3f}] {payload['title']}  ({payload['chunk_id']})")
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    main()
