# ABOUTME: Tests for the knowledge-graph extraction schema (node and relation types).
# ABOUTME: Guards the source/target invariant and that the prompt block exposes every type.

from graphrag_wiki.graph_schema import (
    NODE_TYPES,
    RELATION_TYPES,
    node_types_block,
    schema_prompt_block,
)


def test_node_types_block_lists_nodes_without_relations():
    block = node_types_block()
    for node_type in NODE_TYPES:
        assert node_type in block
    assert "RULED" not in block and "->" not in block


def test_relation_endpoints_are_valid_node_types():
    for name, spec in RELATION_TYPES.items():
        for node_type in spec["sources"] + spec["targets"]:
            assert node_type in NODE_TYPES, f"{name} references unknown node type {node_type}"


def test_prompt_block_names_every_node_and_relation_type():
    block = schema_prompt_block()
    for node_type in NODE_TYPES:
        assert node_type in block
    for relation in RELATION_TYPES:
        assert relation in block
