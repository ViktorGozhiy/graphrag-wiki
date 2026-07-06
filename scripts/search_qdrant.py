# ABOUTME: Semantic (dense vector) search over the Qdrant chunk collection.
# ABOUTME: Embeds the query with the retrieval prefix and prints the top matches with their sources.

import sys

from graphrag_wiki.display import print_hits
from graphrag_wiki.vector import search_vectors

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip() or "How did a Roman emperor gain and hold power?"
    print(f"Q: {query}\n")
    print_hits(search_vectors(query, k=TOP_K))


if __name__ == "__main__":
    main()
