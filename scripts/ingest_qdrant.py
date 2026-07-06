# ABOUTME: Ingests the article corpus into Qdrant — chunk, embed, upsert.
# ABOUTME: Qdrant point ids derive from the stable chunk_id, so re-ingest overwrites rather than duplicates.

import uuid

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from graphrag_wiki.config import (
    CORPUS_PATH,
    EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_SIZE,
)
from graphrag_wiki.corpus import iter_chunks, load_corpus

UPSERT_BATCH = 256


def point_id(chunk_id):
    """Deterministic UUID for a chunk — Qdrant ids must be UUIDs or ints."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def main():
    chunks = list(iter_chunks(load_corpus(CORPUS_PATH)))
    print(f"chunks: {len(chunks)} — embedding with {EMBED_MODEL} (first run downloads the model) ...")

    model = TextEmbedding(model_name=EMBED_MODEL)
    vectors = model.passage_embed(chunk["text"] for chunk in chunks)

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    batch = []
    total = 0
    for chunk, vector in zip(chunks, vectors):
        batch.append(PointStruct(id=point_id(chunk["chunk_id"]), vector=vector.tolist(), payload=chunk))
        if len(batch) >= UPSERT_BATCH:
            client.upsert(QDRANT_COLLECTION, points=batch)
            total += len(batch)
            batch = []
            print(f"  upserted {total}/{len(chunks)}")
    if batch:
        client.upsert(QDRANT_COLLECTION, points=batch)
        total += len(batch)

    print(f"done: {client.count(QDRANT_COLLECTION).count} points in '{QDRANT_COLLECTION}'")


if __name__ == "__main__":
    main()
