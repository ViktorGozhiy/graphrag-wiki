# ABOUTME: Shared OpenAI structured-output helper — one strict-JSON-schema completion with retries.
# ABOUTME: Graph extraction and entity resolution both generate through this so the call lives in one place.

import json

from openai import OpenAI

from graphrag_wiki.config import OPENAI_MODEL

GENERATE_ATTEMPTS = 3

_CLIENT = None


def _client():
    """A lazily-created client so importing this module needs no API key.

    The client retries transient errors (429/5xx) with backoff and caps each attempt
    with a request timeout.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(max_retries=5, timeout=120)
    return _CLIENT


def generate(prompt, schema, name):
    """Return the parsed object from one completion constrained to a strict JSON schema.

    Temperature 0 gives clean, deterministic JSON; a repeat would reproduce a decode
    failure verbatim, so retries warm up slightly to escape a malformed generation.
    """
    error = None
    for attempt in range(GENERATE_ATTEMPTS):
        response = _client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 if attempt == 0 else 0.4,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as decode_error:
            error = decode_error
    raise error
