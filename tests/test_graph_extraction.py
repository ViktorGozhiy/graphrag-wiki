# ABOUTME: Tests for graph-extraction prompt building and validation of triples against the schema.
# ABOUTME: Covers well-formed pass-through and every rejection reason; the LLM call is exercised live.

from graphrag_wiki.graph_extraction import (
    EXTRACTION_FORMAT,
    build_prompt,
    validate,
)
from graphrag_wiki.graph_schema import NODE_TYPES, RELATION_TYPES


def test_build_prompt_embeds_schema_and_passage():
    prompt = build_prompt("Caracalla granted citizenship in 212.")
    assert "Person" in prompt and "ISSUED" in prompt  # schema vocabulary present
    assert "Caracalla granted citizenship in 212." in prompt  # passage present


def test_extraction_format_enums_match_schema_vocabulary():
    props = EXTRACTION_FORMAT["properties"]
    ent_enum = props["entities"]["items"]["properties"]["type"]["enum"]
    rel_enum = props["relations"]["items"]["properties"]["relation"]["enum"]
    assert set(ent_enum) == set(NODE_TYPES)
    assert set(rel_enum) == set(RELATION_TYPES)


def test_validate_keeps_well_formed_entities_and_relations():
    raw = {
        "entities": [
            {"name": "Caracalla", "type": "Person", "description": "An emperor."},
            {"name": "Constitutio Antoniniana", "type": "Law", "description": "An edict."},
        ],
        "relations": [
            {
                "source": "Caracalla",
                "relation": "ISSUED",
                "target": "Constitutio Antoniniana",
                "description": "He issued it.",
            },
        ],
    }
    entities, relations, rejects = validate(raw)
    assert [e["name"] for e in entities] == ["Caracalla", "Constitutio Antoniniana"]
    assert len(relations) == 1
    assert rejects == []


def test_validate_rejects_unknown_entity_type_and_its_dependent_relation():
    raw = {
        "entities": [
            {"name": "Caracalla", "type": "Emperor", "description": "..."},  # Emperor is not a node type
            {"name": "Rome", "type": "Place", "description": "..."},
        ],
        "relations": [
            {"source": "Caracalla", "relation": "RULED", "target": "Rome", "description": "..."},
        ],
    }
    entities, relations, rejects = validate(raw)
    assert [e["name"] for e in entities] == ["Rome"]
    assert relations == []  # RULED loses its source entity, so it dangles
    assert any("unknown node type" in reason for _, _, reason in rejects)
    assert any("endpoint not declared" in reason for _, _, reason in rejects)


def test_validate_rejects_unknown_relation_type():
    raw = {
        "entities": [
            {"name": "Caracalla", "type": "Person", "description": "..."},
            {"name": "Rome", "type": "Place", "description": "..."},
        ],
        "relations": [
            {"source": "Caracalla", "relation": "CONQUERED", "target": "Rome", "description": "..."},
        ],
    }
    _, relations, rejects = validate(raw)
    assert relations == []
    assert any("unknown relation type" in reason for _, _, reason in rejects)


def test_validate_rejects_relation_with_wrong_endpoint_types():
    # MEMBER_OF must target an Organization, not a Place.
    raw = {
        "entities": [
            {"name": "Cicero", "type": "Person", "description": "..."},
            {"name": "Rome", "type": "Place", "description": "..."},
        ],
        "relations": [
            {"source": "Cicero", "relation": "MEMBER_OF", "target": "Rome", "description": "..."},
        ],
    }
    _, relations, rejects = validate(raw)
    assert relations == []
    assert any("endpoint type violates schema" in reason for _, _, reason in rejects)


def test_validate_rejects_relation_referencing_undeclared_entity():
    raw = {
        "entities": [{"name": "Caracalla", "type": "Person", "description": "..."}],
        "relations": [
            {"source": "Caracalla", "relation": "RULED", "target": "Gaul", "description": "..."},
        ],
    }
    _, relations, rejects = validate(raw)
    assert relations == []
    assert any("endpoint not declared" in reason for _, _, reason in rejects)
