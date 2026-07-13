# ABOUTME: Tests for graph-aware retrieval — linking a question to graph nodes and traversing the neighborhood.
# ABOUTME: Covers capitalized-span extraction, name-index matching, and SAME_AS-expanded traversal.

import pytest

from graphrag_wiki import graph_store
from graphrag_wiki.graph_retrieval import (
    capitalized_spans,
    graph_chunk_ids,
    link_spans,
    neighborhood,
    rank_chunks,
)

PFX = "Ztest "


@pytest.fixture
def session():
    driver = graph_store.driver()
    driver.verify_connectivity()
    with driver.session() as sess:
        sess.run("MATCH (n:Entity) WHERE n.name STARTS WITH $pfx DETACH DELETE n", pfx=PFX)
        yield sess
        sess.run("MATCH (n:Entity) WHERE n.name STARTS WITH $pfx DETACH DELETE n", pfx=PFX)
    driver.close()


def _record(chunk_id, entities, relations):
    return {"chunk_id": chunk_id, "title": "T", "entities": entities, "relations": relations}


def _ent(name, type_):
    return {"name": PFX + name, "type": type_, "description": "d"}


def _rel(source, relation, target):
    return {"source": PFX + source, "relation": relation, "target": PFX + target, "description": relation}


def test_capitalized_spans_extracts_consecutive_capitalized_runs():
    spans = capitalized_spans("How did Emperor Caracalla issue the Antonine Constitution?")
    assert spans == ["How", "Emperor Caracalla", "Antonine Constitution"]


def test_capitalized_spans_lowercase_words_break_runs():
    assert capitalized_spans("the Roman Empire and its provinces") == ["Roman Empire"]


def test_capitalized_spans_none_when_no_capitals():
    assert capitalized_spans("how did citizenship affect taxation?") == []


def test_link_spans_matches_by_content_tokens_ignoring_honorifics():
    nodes = [("Caracalla", "Person"), ("Antonine Constitution", "Law"), ("citizenship", "Concept")]
    spans = ["Emperor Caracalla", "Antonine Constitution"]
    assert link_spans(spans, nodes) == {("Caracalla", "Person"), ("Antonine Constitution", "Law")}


def test_link_spans_returns_all_nodes_sharing_a_name():
    nodes = [("Rome", "Place"), ("Rome", "Organization")]
    assert link_spans(["Rome"], nodes) == {("Rome", "Place"), ("Rome", "Organization")}


def test_link_spans_ignores_unmatched_spans():
    assert link_spans(["How", "Nero"], [("Caracalla", "Person")]) == set()


def test_rank_chunks_orders_by_nearest_hop_then_edge_frequency():
    edges = [
        {"hop": 2, "chunk_ids": ["far"]},
        {"hop": 1, "chunk_ids": ["near", "shared"]},
        {"hop": 1, "chunk_ids": ["shared"]},
    ]
    # shared: hop 1, on 2 edges; near: hop 1, on 1 edge; far: hop 2
    assert rank_chunks(edges) == ["shared", "near", "far"]


def test_neighborhood_collects_one_hop_edge_with_provenance(session):
    graph_store.load_records(
        [_record(
            "71413044:3",
            [_ent("Caracalla", "Person"), _ent("citizenship", "Concept")],
            [_rel("Caracalla", "GRANTED", "citizenship")],
        )],
        session,
    )
    edges = neighborhood([(PFX + "Caracalla", "Person")], session, hops=1)
    assert edges == [{
        "relation": "GRANTED",
        "source": PFX + "Caracalla", "source_type": "Person",
        "target": PFX + "citizenship", "target_type": "Concept",
        "descriptions": ["GRANTED"],
        "chunk_ids": ["71413044:3"],
        "hop": 1,
    }]


def test_neighborhood_expands_same_as_so_variants_act_as_one(session):
    graph_store.load_records(
        [
            _record("c0", [_ent("Caracalla", "Person"), _ent("Emperor Caracalla", "Person")], []),
            _record(
                "c1",
                [_ent("Emperor Caracalla", "Person"), _ent("Severan dynasty", "Organization")],
                [_rel("Emperor Caracalla", "MEMBER_OF", "Severan dynasty")],
            ),
        ],
        session,
    )
    graph_store.link_same_as(
        [{
            "a": {"name": PFX + "Caracalla", "type": "Person"},
            "b": {"name": PFX + "Emperor Caracalla", "type": "Person"},
            "canonical": PFX + "Caracalla",
        }],
        session,
    )
    edges = neighborhood([(PFX + "Caracalla", "Person")], session, hops=1)
    assert ("MEMBER_OF", PFX + "Severan dynasty") in {(e["relation"], e["target"]) for e in edges}


def test_neighborhood_walks_multiple_hops_tagging_distance(session):
    graph_store.load_records(
        [
            _record("c0", [_ent("A", "Person"), _ent("B", "Organization")], [_rel("A", "MEMBER_OF", "B")]),
            _record("c1", [_ent("B", "Organization"), _ent("C", "Place")], [_rel("B", "LOCATED_IN", "C")]),
        ],
        session,
    )
    assert {e["target"] for e in neighborhood([(PFX + "A", "Person")], session, hops=1)} == {PFX + "B"}

    reached = {(e["hop"], e["relation"], e["target"]) for e in neighborhood([(PFX + "A", "Person")], session, hops=2)}
    assert (1, "MEMBER_OF", PFX + "B") in reached
    assert (2, "LOCATED_IN", PFX + "C") in reached


def test_neighborhood_caps_edges_per_node_beyond_the_first_hop(session):
    records = [_record("c0", [_ent("A", "Person"), _ent("B", "Organization")], [_rel("A", "MEMBER_OF", "B")])]
    for i in range(5):
        records.append(
            _record(f"c{i + 1}", [_ent("B", "Organization"), _ent(f"P{i}", "Place")], [_rel("B", "LOCATED_IN", f"P{i}")])
        )
    graph_store.load_records(records, session)

    edges = neighborhood([(PFX + "A", "Person")], session, hops=2, cap=2)
    assert len([e for e in edges if e["hop"] == 1]) == 1  # the single seed edge is never capped
    assert len([e for e in edges if e["hop"] == 2]) == 2  # B's five neighbors capped to two


def test_graph_chunk_ids_links_query_traverses_and_ranks_bridge_first(session):
    graph_store.load_records(
        [
            _record(
                "bridge",
                [_ent("Caracalla", "Person"), _ent("citizenship", "Concept")],
                [_rel("Caracalla", "GRANTED", "citizenship")],
            ),
            _record(
                "far",
                [_ent("citizenship", "Concept"), _ent("Taxation", "Concept")],
                [_rel("citizenship", "INFLUENCED", "Taxation")],
            ),
        ],
        session,
    )
    ids = graph_chunk_ids(f"How did {PFX}Caracalla and citizenship affect taxation?", session, hops=2)
    assert ids[0] == "bridge"  # the hop-1 edge's provenance outranks the hop-2 chunk
    assert "far" in ids  # the hop-2 chunk is still reached
