# ABOUTME: In-memory BM25 keyword index over the shared chunk records.
# ABOUTME: Transparent lexical retrieval to compare against, and later fuse with, vector search.

import re

from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\w+")


def tokenize(text):
    """Lowercase and split text into word tokens."""
    return TOKEN.findall(text.lower())


class BM25Index:
    """BM25 ranking over chunk records, keyed by their text field."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])

    def search(self, query, k=5):
        """Return the top-k chunk records for a query, each with a BM25 score."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [{**chunk, "score": float(score)} for chunk, score in ranked[:k]]
