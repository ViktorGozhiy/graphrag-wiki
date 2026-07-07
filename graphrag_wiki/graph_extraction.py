# ABOUTME: Extracts typed entities and relations from a chunk via a schema-constrained LLM call.
# ABOUTME: Builds the prompt, enforces the schema vocabulary in the JSON output, and validates triples.

import json

import requests

from graphrag_wiki.config import OLLAMA_MODEL, OLLAMA_URL
from graphrag_wiki.graph_schema import NODE_TYPES, RELATION_TYPES, schema_prompt_block

PROMPT_TEMPLATE = (
    "You extract a knowledge graph from a passage about ancient Rome. Use ONLY "
    "the node and relation types listed below. If a relation does not fit one of "
    "the listed types, omit it — never invent a type.\n\n"
    "{schema}\n\n"
    "List as entities everything you will connect, including abstract concepts "
    "(institutions, rights, practices, ideas) such as citizenship or taxation. "
    "For each entity give: name (as it appears in the passage), type (from the "
    "list), and a one-sentence description grounded in the passage.\n\n"
    "For each relation give: source, relation (from the list), target, and a "
    "one-sentence description. The source and target MUST both be names you "
    "listed in entities — if either is missing, add it as an entity or omit the "
    "relation. Use MEMBER_OF when a person belongs to a body or class; use "
    "PART_OF only when a body or place sits inside a larger body or place.\n\n"
    "Passage:\n{text}\n"
)

EXTRACTION_FORMAT = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(NODE_TYPES)},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string", "enum": list(RELATION_TYPES)},
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "relation", "target", "description"],
            },
        },
    },
    "required": ["entities", "relations"],
}


def build_prompt(text):
    return PROMPT_TEMPLATE.format(schema=schema_prompt_block(), text=text)


def validate(raw):
    """Filter extracted entities and relations down to those the schema allows.

    Returns (entities, relations, rejects). Each reject is a (kind, item, reason)
    tuple so a batch run can report what the extractor got wrong.
    """
    rejects = []
    entities = []
    entity_type = {}
    for entity in raw.get("entities", []):
        name = entity.get("name", "").strip()
        node_type = entity.get("type")
        if not name:
            rejects.append(("entity", entity, "missing name"))
            continue
        if node_type not in NODE_TYPES:
            rejects.append(("entity", entity, "unknown node type"))
            continue
        entities.append(entity)
        entity_type[name] = node_type

    relations = []
    for relation in raw.get("relations", []):
        rel_type = relation.get("relation")
        source = relation.get("source", "").strip()
        target = relation.get("target", "").strip()
        if rel_type not in RELATION_TYPES:
            rejects.append(("relation", relation, "unknown relation type"))
            continue
        if source not in entity_type or target not in entity_type:
            rejects.append(("relation", relation, "endpoint not declared"))
            continue
        spec = RELATION_TYPES[rel_type]
        if entity_type[source] not in spec["sources"] or entity_type[target] not in spec["targets"]:
            rejects.append(("relation", relation, "endpoint type violates schema"))
            continue
        relations.append(relation)

    return entities, relations, rejects


def extract(chunk_id, text):
    """Extract validated entities and relations from one chunk of text."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": build_prompt(text),
            "format": EXTRACTION_FORMAT,
            "stream": False,
        },
        timeout=900,
    )
    response.raise_for_status()
    raw = json.loads(response.json()["response"])
    entities, relations, rejects = validate(raw)
    return {"chunk_id": chunk_id, "entities": entities, "relations": relations, "rejects": rejects}
