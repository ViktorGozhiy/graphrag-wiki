# ABOUTME: Loads extraction records into Neo4j as :Entity nodes and typed relationship edges.
# ABOUTME: Nodes MERGE on (name, type); provenance chunk_ids and edge descriptions accumulate without duplicates.

from neo4j import GraphDatabase

from graphrag_wiki.config import NEO4J_AUTH, NEO4J_URL
from graphrag_wiki.graph_schema import RELATION_TYPES

_MERGE_NODE = (
    "MERGE (n:Entity {name: $name, type: $type}) "
    "SET n.chunk_ids = CASE WHEN $chunk_id IN coalesce(n.chunk_ids, []) "
    "THEN n.chunk_ids ELSE coalesce(n.chunk_ids, []) + $chunk_id END"
)


def _merge_edge(relation):
    """Cypher to upsert one typed edge; relation is whitelisted against the closed schema."""
    return (
        "MERGE (s:Entity {name: $source, type: $source_type}) "
        "MERGE (t:Entity {name: $target, type: $target_type}) "
        f"MERGE (s)-[r:{relation}]->(t) "
        "SET r.chunk_ids = CASE WHEN $chunk_id IN coalesce(r.chunk_ids, []) "
        "THEN r.chunk_ids ELSE coalesce(r.chunk_ids, []) + $chunk_id END, "
        "r.descriptions = CASE WHEN $description IN coalesce(r.descriptions, []) "
        "THEN r.descriptions ELSE coalesce(r.descriptions, []) + $description END"
    )


def driver():
    """Open a Neo4j driver from config."""
    return GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH)


def ensure_indexes(session):
    """Create the composite index MERGE relies on so node lookups stay fast."""
    session.run("CREATE INDEX entity_name_type IF NOT EXISTS FOR (n:Entity) ON (n.name, n.type)")


_LINK_SAME_AS = (
    "MATCH (s:Entity {name: $source_name, type: $source_type}) "
    "MATCH (t:Entity {name: $target_name, type: $target_type}) "
    "MERGE (s)-[r:SAME_AS]->(t) SET r.canonical = $canonical"
)


def link_same_as(links, session):
    """Link resolved duplicates with a SAME_AS edge carrying the chosen canonical name.

    Each link is {"a": {name, type}, "b": {name, type}, "canonical": name}. Endpoints are
    ordered before MERGE so the edge is stored once regardless of pair order — re-running
    updates the same edge rather than adding a mirror. Returns the number of links written.
    """
    for link in links:
        source, target = sorted(
            [(link["a"]["name"], link["a"]["type"]), (link["b"]["name"], link["b"]["type"])]
        )
        session.run(
            _LINK_SAME_AS,
            source_name=source[0],
            source_type=source[1],
            target_name=target[0],
            target_type=target[1],
            canonical=link["canonical"],
        )
    return len(links)


def load_records(records, session):
    """Upsert entities as typed nodes and relations as typed edges, accumulating chunk_id provenance.

    Edge endpoint types are resolved from the same record's entities (relations carry only names).
    Returns counts of what was loaded and skipped.
    """
    ensure_indexes(session)
    counts = {"entities": 0, "relations": 0, "skipped_relations": 0}
    for record in records:
        chunk_id = record["chunk_id"]
        name_type = {entity["name"]: entity["type"] for entity in record["entities"]}
        for entity in record["entities"]:
            session.run(_MERGE_NODE, name=entity["name"], type=entity["type"], chunk_id=chunk_id)
            counts["entities"] += 1
        for relation in record["relations"]:
            source_type = name_type.get(relation["source"])
            target_type = name_type.get(relation["target"])
            if relation["relation"] not in RELATION_TYPES or source_type is None or target_type is None:
                counts["skipped_relations"] += 1
                continue
            session.run(
                _merge_edge(relation["relation"]),
                source=relation["source"],
                source_type=source_type,
                target=relation["target"],
                target_type=target_type,
                chunk_id=chunk_id,
                description=relation["description"],
            )
            counts["relations"] += 1
    return counts
