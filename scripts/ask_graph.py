# ABOUTME: Side-by-side demo — answers a question with baseline hybrid RAG and with graph-augmented RAG.
# ABOUTME: The graph adds a reserved block of entity-connected chunks so a bridge passage reaches the model.

import sys

from graphrag_wiki.generation import answer
from graphrag_wiki.graph_retrieval import graph_passages
from graphrag_wiki.graph_store import driver
from graphrag_wiki.hybrid import search_hybrid

TOP_K = 5
GRAPH_M = 6

DEFAULT_QUESTION = (
    "Which emperor granted Roman citizenship to all free inhabitants of the empire, "
    "and how did that grant affect Roman taxation?"
)

INSTRUCTION = (
    "You answer questions about ancient Rome using ONLY the numbered context below. "
    "If the answer is not in the context, say you don't know. Cite the sources you rely "
    "on by their [n] number.\n\n"
)

BASELINE_PROMPT = INSTRUCTION + "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

GRAPH_PROMPT = (
    INSTRUCTION
    + "Retrieved passages:\n{passages}\n\n"
    + "Connected via knowledge graph:\n{graph}\n\n"
    + "Question: {question}\n\nAnswer:"
)


def numbered(hits, start=1):
    """Render chunk records as a numbered context block starting at `start`."""
    return "\n\n".join(f"[{i}] ({hit['title']}) {hit['text']}" for i, hit in enumerate(hits, start))


def sources(hits, start=1, mark=None):
    lines = []
    for i, hit in enumerate(hits, start):
        flag = "  <- graph-only" if mark and hit["chunk_id"] in mark else ""
        lines.append(f"  [{i}] {hit['title']} ({hit['chunk_id']}){flag}")
    return "\n".join(lines)


def main():
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    print(f"Q: {question}\n")

    hybrid_hits = search_hybrid(question, k=TOP_K)
    with driver().session() as session:
        graph_hits = graph_passages(question, session, top_m=GRAPH_M)
    # The graph earns its place by reaching articles the flat retrieval missed; re-adding chunks
    # from articles hybrid already covered only dilutes the context, so keep only new articles.
    seen_titles = {hit["title"] for hit in hybrid_hits}
    graph_only = [hit for hit in graph_hits if hit["title"] not in seen_titles]

    baseline = answer(BASELINE_PROMPT.format(context=numbered(hybrid_hits), question=question))
    graph = answer(
        GRAPH_PROMPT.format(
            passages=numbered(hybrid_hits),
            graph=numbered(graph_only, start=len(hybrid_hits) + 1),
            question=question,
        )
    )

    print("=== BASELINE (hybrid only) ===")
    print(baseline.strip())
    print("\nSources:")
    print(sources(hybrid_hits))

    print("\n=== GRAPHRAG (hybrid + graph block) ===")
    print(graph.strip())
    print("\nSources:")
    print(sources(hybrid_hits))
    print(sources(graph_only, start=len(hybrid_hits) + 1, mark={hit["chunk_id"] for hit in graph_only}))


if __name__ == "__main__":
    main()
