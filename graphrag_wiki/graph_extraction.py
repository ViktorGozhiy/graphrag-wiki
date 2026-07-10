# ABOUTME: Extracts typed entities then relations from a chunk via two schema-constrained LLM calls.
# ABOUTME: Pass two restricts relation endpoints to the entities pass one declared, then validates triples.

from graphrag_wiki import llm
from graphrag_wiki.graph_schema import (
    NODE_TYPES,
    RELATION_TYPES,
    node_types_block,
    schema_prompt_block,
)

ENTITY_PROMPT_TEMPLATE = (
    "You extract the entities of a knowledge graph from a passage about ancient "
    "Rome. Use ONLY these node types.\n\n"
    "{nodes}\n\n"
    "List every entity that matters, including abstract concepts (institutions, "
    "rights, practices, ideas) such as citizenship or taxation. For each give: "
    "name (as it appears in the passage), type (from the list), and a one-sentence "
    "description grounded in the passage.\n\n"
    "Passage:\n{text}\n"
)

RELATION_PROMPT_TEMPLATE = (
    "You extract the relations of a knowledge graph from a passage about ancient "
    "Rome. Use ONLY the relation types below, and connect ONLY the listed "
    "entities.\n\n"
    "{schema}\n\n"
    "Entities (name — type):\n{entities}\n\n"
    "For each relation give: source and target (both from the entity list), "
    "relation (from the list), and a one-sentence description. Add a relation "
    "only when the source and target types match one of the allowed directions "
    "above; otherwise omit it.\n\n"
    "Passage:\n{text}\n"
)

ENTITY_FORMAT = {
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
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def relation_format(entity_names):
    """JSON schema for pass two: endpoints are constrained to the declared entities."""
    return {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": entity_names},
                        "relation": {"type": "string", "enum": list(RELATION_TYPES)},
                        "target": {"type": "string", "enum": entity_names},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "relation", "target", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["relations"],
        "additionalProperties": False,
    }


def build_entity_prompt(text):
    return ENTITY_PROMPT_TEMPLATE.format(nodes=node_types_block(), text=text)


def build_relation_prompt(text, entities):
    listing = "\n".join(f"- {e['name']} — {e['type']}" for e in entities)
    return RELATION_PROMPT_TEMPLATE.format(schema=schema_prompt_block(), entities=listing, text=text)


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


def sanitize_entities(entities):
    """Strip characters a strict-mode schema enum forbids (a double quote) from entity names.

    The names become relation_format's source/target enum, so a raw " in a name makes the
    pass-two schema invalid. Cleaning here keeps the names, that enum, the pass-two prompt,
    and validation all consistent.
    """
    for entity in entities:
        entity["name"] = entity.get("name", "").replace('"', "").strip()
    return entities


def extract(chunk_id, text):
    """Extract validated entities and relations from one chunk in two passes."""
    raw_entities = llm.generate(build_entity_prompt(text), ENTITY_FORMAT, "graph_extraction")
    entities = sanitize_entities(raw_entities.get("entities", []))
    names = list(dict.fromkeys(e["name"].strip() for e in entities if e.get("name", "").strip()))

    relations = []
    if names:
        prompt = build_relation_prompt(text, entities)
        raw_relations = llm.generate(prompt, relation_format(names), "graph_extraction")
        relations = raw_relations.get("relations", [])

    entities, relations, rejects = validate({"entities": entities, "relations": relations})
    return {"chunk_id": chunk_id, "entities": entities, "relations": relations, "rejects": rejects}
