# ABOUTME: Tests for the recursive text chunker shared across all retrievers.
# ABOUTME: Verifies size bounds, paragraph/sentence boundaries, overlap, and no content loss.

from graphrag_wiki.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=20) == []


def test_short_text_is_single_chunk():
    text = "A short paragraph about Rome."
    assert chunk_text(text, chunk_size=100, overlap=20) == [text]


def test_long_text_splits_into_bounded_chunks():
    paragraphs = [f"Paragraph {i} " + "word " * 20 for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        # bound is chunk_size + overlap, plus the 2-char paragraph separator
        assert len(chunk) <= 100 + 20 + 2


def test_consecutive_chunks_overlap():
    paragraphs = [f"Para{i} " + "alpha " * 15 for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) >= 2
    tail_of_first = chunks[0][-15:]
    assert tail_of_first in chunks[1]


def test_long_single_paragraph_is_split_by_sentences():
    text = ". ".join(f"Sentence {i} about the empire" for i in range(20)) + "."
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100 + 20 + 2


def test_no_paragraph_content_is_lost():
    paragraphs = [f"UNIQUE{i}marker " + "filler " * 10 for i in range(8)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    joined = " ".join(chunks)
    for i in range(8):
        assert f"UNIQUE{i}marker" in joined
