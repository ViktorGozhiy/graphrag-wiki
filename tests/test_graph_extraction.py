# ABOUTME: Tests for graph-extraction prompt building and validation of triples against the schema.
# ABOUTME: Covers well-formed pass-through and every rejection reason; the LLM call is exercised live.

from graphrag_wiki.graph_extraction import (
    build_entity_prompt,
    build_relation_prompt,
    relation_format,
    validate,
)
from graphrag_wiki.graph_schema import RELATION_TYPES


def test_entity_prompt_embeds_node_types_and_passage():
    prompt = build_entity_prompt("Caracalla granted citizenship in 212.")
    assert "Person" in prompt and "Concept" in prompt  # node vocabulary present
    assert "Caracalla granted citizenship in 212." in prompt  # passage present


def test_relation_prompt_lists_entities_with_types_schema_and_passage():
    entities = [
        {"name": "Caracalla", "type": "Person", "description": "..."},
        {"name": "citizenship", "type": "Concept", "description": "..."},
    ]
    prompt = build_relation_prompt("Caracalla granted citizenship.", entities)
    assert "Caracalla" in prompt and "citizenship" in prompt  # the fixed entity list
    assert "Person" in prompt and "Concept" in prompt  # their types
    assert "GRANTED" in prompt  # relation schema present
    assert "Caracalla granted citizenship." in prompt  # passage present


def test_relation_format_constrains_endpoints_to_given_entities():
    fmt = relation_format(["Caracalla", "citizenship"])
    item = fmt["properties"]["relations"]["items"]["properties"]
    assert item["source"]["enum"] == ["Caracalla", "citizenship"]
    assert item["target"]["enum"] == ["Caracalla", "citizenship"]
    assert set(item["relation"]["enum"]) == set(RELATION_TYPES)


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


def test_validate_accepts_organization_as_ruler_of_a_place():
    raw = {
        "entities": [
            {"name": "Roman Empire", "type": "Organization", "description": "..."},
            {"name": "Britain", "type": "Place", "description": "..."},
        ],
        "relations": [
            {"source": "Roman Empire", "relation": "RULED", "target": "Britain", "description": "..."},
        ],
    }
    _, relations, rejects = validate(raw)
    assert [r["relation"] for r in relations] == ["RULED"]
    assert rejects == []


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
