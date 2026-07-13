# ABOUTME: Grounded answer generation with the local Ollama model.
# ABOUTME: Sends one prompt and returns the full completion for scripts that compare answers.

import requests

from graphrag_wiki.config import OLLAMA_MODEL, OLLAMA_URL


def answer(prompt):
    """Return the local model's completion for a grounded prompt."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=900,
    )
    response.raise_for_status()
    return response.json()["response"]
