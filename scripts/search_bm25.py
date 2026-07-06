# ABOUTME: Keyword (BM25) search over the article corpus chunks.
# ABOUTME: Builds an in-memory BM25 index and prints the top matches with their sources.

import sys

from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.display import print_hits
from graphrag_wiki.keyword import BM25Index

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip() or "the denarius and Roman coinage"
    index = BM25Index(iter_chunks(load_corpus(CORPUS_PATH)))
    print(f"Q: {query}\n")
    print_hits(index.search(query, k=TOP_K))


if __name__ == "__main__":
    main()
