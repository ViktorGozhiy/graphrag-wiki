# ABOUTME: Retrieval-augmented generation — retrieve hybrid context, ask the local LLM, cite sources.
# ABOUTME: Grounds the Ollama model on the retrieved chunks and streams a cited answer.

import json
import sys

import requests

from graphrag_wiki.config import OLLAMA_MODEL, OLLAMA_URL
from graphrag_wiki.hybrid import search_hybrid

TOP_K = 5

PROMPT_TEMPLATE = (
    "You answer questions about ancient Rome using ONLY the numbered context "
    "below. If the answer is not in the context, say you don't know. Cite the "
    "sources you rely on by their [n] number.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_context(hits):
    return "\n\n".join(f"[{i}] ({hit['title']}) {hit['text']}" for i, hit in enumerate(hits, 1))


def main():
    question = " ".join(sys.argv[1:]).strip() or "How did a Roman emperor gain and hold power?"
    hits = search_hybrid(question, k=TOP_K)
    prompt = PROMPT_TEMPLATE.format(context=build_context(hits), question=question)

    print(f"Q: {question}\n")
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        stream=True,
        timeout=900,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            print(event.get("response", ""), end="", flush=True)
            if event.get("done"):
                break

    print("\n\nSources:")
    for index, hit in enumerate(hits, 1):
        print(f"  [{index}] {hit['title']} ({hit['chunk_id']}) — {hit['url']}")


if __name__ == "__main__":
    main()
