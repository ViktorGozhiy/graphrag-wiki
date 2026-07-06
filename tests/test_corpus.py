# ABOUTME: Tests for turning corpus articles into the shared chunk records.
# ABOUTME: Verifies stable per-article chunk ids and carried-over metadata.

from graphrag_wiki.corpus import iter_chunks


def test_iter_chunks_assigns_sequential_ids_per_article():
    rows = [
        {"pageid": 42, "title": "Alpha", "url": "u1", "text": "Short para.\n\n" + "long " * 200},
        {"pageid": 7, "title": "Beta", "url": "u2", "text": "Only one paragraph here."},
    ]
    chunks = list(iter_chunks(rows, chunk_size=100, overlap=20))
    alpha = [c for c in chunks if c["pageid"] == 42]
    beta = [c for c in chunks if c["pageid"] == 7]

    assert [c["chunk_id"] for c in alpha[:2]] == ["42:0", "42:1"]
    assert alpha[0]["chunk_index"] == 0
    assert beta[0]["chunk_id"] == "7:0"
    assert all(c["title"] and c["url"] and c["text"] for c in chunks)
