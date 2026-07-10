# ABOUTME: Tests for loading extraction records into Neo4j as typed nodes and edges.
# ABOUTME: Runs against a live Neo4j; test entities are name-prefixed and cleaned up around each test.

import pytest

from graphrag_wiki import graph_store

PFX = "zzz_test_"


def _record(chunk_id, entities, relations):
    return {"chunk_id": chunk_id, "title": "T", "entities": entities, "relations": relations}


def _ent(name, type_, description="d"):
    return {"name": PFX + name, "type": type_, "description": description}


def _rel(source, relation, target, description="d"):
    return {"source": PFX + source, "relation": relation, "target": PFX + target, "description": description}


@pytest.fixture
def session():
    driver = graph_store.driver()
    driver.verify_connectivity()
    with driver.session() as sess:
        sess.run("MATCH (n:Entity) WHERE n.name STARTS WITH $pfx DETACH DELETE n", pfx=PFX)
        yield sess
        sess.run("MATCH (n:Entity) WHERE n.name STARTS WITH $pfx DETACH DELETE n", pfx=PFX)
    driver.close()


def _node(session, name):
    rec = session.run(
        "MATCH (n:Entity {name:$name}) RETURN n.type AS type, n.chunk_ids AS chunk_ids",
        name=PFX + name,
    ).single()
    return rec and {"type": rec["type"], "chunk_ids": set(rec["chunk_ids"])}


def test_load_creates_typed_nodes_and_edge(session):
    record = _record(
        "1:0",
        [_ent("Caracalla", "Person"), _ent("citizenship", "Concept")],
        [_rel("Caracalla", "GRANTED", "citizenship", "Caracalla granted universal citizenship.")],
    )
    graph_store.load_records([record], session)

    assert _node(session, "Caracalla") == {"type": "Person", "chunk_ids": {"1:0"}}
    assert _node(session, "citizenship") == {"type": "Concept", "chunk_ids": {"1:0"}}

    edge = session.run(
        "MATCH (:Entity {name:$s})-[r:GRANTED]->(:Entity {name:$t}) "
        "RETURN r.chunk_ids AS chunk_ids, r.descriptions AS descriptions",
        s=PFX + "Caracalla",
        t=PFX + "citizenship",
    ).single()
    assert set(edge["chunk_ids"]) == {"1:0"}
    assert edge["descriptions"] == ["Caracalla granted universal citizenship."]


def test_same_name_different_type_are_distinct_nodes(session):
    records = [
        _record("1:0", [_ent("Roman Empire", "Place")], []),
        _record("2:0", [_ent("Roman Empire", "Organization")], []),
    ]
    graph_store.load_records(records, session)

    count = session.run(
        "MATCH (n:Entity {name:$name}) RETURN count(n) AS c", name=PFX + "Roman Empire"
    ).single()["c"]
    assert count == 2


def test_idempotent_and_provenance_accumulates(session):
    records = [
        _record("1:0", [_ent("Rome", "Place")], []),
        _record("2:0", [_ent("Rome", "Place")], []),
    ]
    graph_store.load_records(records, session)
    graph_store.load_records(records, session)  # re-load must not duplicate

    count = session.run(
        "MATCH (n:Entity {name:$name}) RETURN count(n) AS c", name=PFX + "Rome"
    ).single()["c"]
    assert count == 1
    assert _node(session, "Rome") == {"type": "Place", "chunk_ids": {"1:0", "2:0"}}


def test_link_same_as_is_a_single_idempotent_edge_carrying_the_canonical(session):
    graph_store.load_records(
        [_record("1:0", [_ent("Caracalla", "Person"), _ent("Emperor Caracalla", "Person")], [])],
        session,
    )
    link = {
        "a": {"name": PFX + "Caracalla", "type": "Person"},
        "b": {"name": PFX + "Emperor Caracalla", "type": "Person"},
        "canonical": PFX + "Caracalla",
    }
    graph_store.link_same_as([link], session)
    # re-linking, even with endpoints reversed, must not create a second edge
    graph_store.link_same_as([{"a": link["b"], "b": link["a"], "canonical": link["canonical"]}], session)

    rows = session.run(
        "MATCH (:Entity {name:$a})-[r:SAME_AS]-(:Entity {name:$b}) RETURN r.canonical AS canonical",
        a=PFX + "Caracalla",
        b=PFX + "Emperor Caracalla",
    ).data()
    assert len(rows) == 1
    assert rows[0]["canonical"] == PFX + "Caracalla"
