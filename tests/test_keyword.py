# ABOUTME: Tests for the in-memory BM25 keyword index.
# ABOUTME: Verifies tokenization and that exact term matches rank first.

from graphrag_wiki.keyword import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_on_non_word():
    assert tokenize("The Denarius, a coin!") == ["the", "denarius", "a", "coin"]


def test_bm25_ranks_exact_term_match_first():
    chunks = [
        {"chunk_id": "1:0", "title": "A", "text": "The Roman senate met in the forum."},
        {"chunk_id": "2:0", "title": "B", "text": "Gladiators fought in the arena."},
        {"chunk_id": "3:0", "title": "C", "text": "Emperors issued a denarius, a silver coin."},
    ]
    index = BM25Index(chunks)
    hits = index.search("denarius coin", k=2)

    assert hits[0]["chunk_id"] == "3:0"
    assert "score" in hits[0]
    assert len(hits) == 2
