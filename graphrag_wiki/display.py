# ABOUTME: Shared terminal formatting for retrieval results.
# ABOUTME: Keeps vector, keyword, and comparison scripts printing hits the same way.

def print_hits(hits, indent=""):
    """Print ranked chunk hits as score, source, and a text snippet."""
    for hit in hits:
        snippet = " ".join(hit["text"].split())[:200]
        print(f"{indent}[{hit['score']:.3f}] {hit['title']}  ({hit['chunk_id']})")
        print(f"{indent}    {snippet}...")
