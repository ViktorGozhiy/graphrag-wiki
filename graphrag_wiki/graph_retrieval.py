# ABOUTME: Graph-aware retrieval — links a question to graph nodes and gathers evidence from their neighborhood.
# ABOUTME: Seeds from capitalized spans in the question, then traverses SAME_AS-expanded edges for bridge chunks.

import collections
import re

from graphrag_wiki.entity_resolution import name_tokens

_SPAN = re.compile(r"[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*")


def capitalized_spans(query):
    """Return maximal runs of consecutive capitalized words in the query.

    Sentence-initial words ("How") are included; they simply fail to match any node
    when the spans are looked up against the graph, so no stopword list is needed.
    """
    return _SPAN.findall(query)


def link_spans(spans, nodes):
    """Return the (name, type) nodes whose content tokens match a span in the query.

    nodes is a list of (name, type). Both sides are keyed by honorific-stripped content
    tokens, so "Emperor Caracalla" links to the node "Caracalla". A name shared by nodes
    of different types links all of them; spans matching nothing are dropped.
    """
    index = collections.defaultdict(list)
    for name, node_type in nodes:
        tokens = name_tokens(name)
        if tokens:
            index[tokens].append((name, node_type))

    matched = set()
    for span in spans:
        matched.update(index.get(name_tokens(span), []))
    return matched


# Typed edges of a node and all its SAME_AS variants (variants act as one logical node);
# SAME_AS itself is never a traversal step. startNode/endNode preserve the stored direction
# so descriptions read correctly, while `other` is the endpoint to continue walking from.
_EDGES = (
    "MATCH (n:Entity {name: $name, type: $type})-[:SAME_AS*0..]-(self:Entity) "
    "MATCH (self)-[r]-(m:Entity) WHERE type(r) <> 'SAME_AS' "
    "RETURN type(r) AS relation, "
    "startNode(r).name AS source, startNode(r).type AS source_type, "
    "endNode(r).name AS target, endNode(r).type AS target_type, "
    "coalesce(r.descriptions, []) AS descriptions, coalesce(r.chunk_ids, []) AS chunk_ids, "
    "m.name AS other_name, m.type AS other_type"
)


def neighborhood(seeds, session, hops=2, cap=10):
    """Walk typed edges out from the seed nodes and return the edges reached, tagged by hop.

    Seeds and every node they reach are expanded across SAME_AS so name variants act as one.
    Each edge carries its provenance chunk_ids and descriptions. Traversal is breadth-first;
    an edge is returned once, at the shallowest hop it is reached.
    """
    frontier = set(seeds)
    visited = set(frontier)
    edges = []
    seen = set()
    for hop in range(1, hops + 1):
        next_frontier = set()
        for name, node_type in frontier:
            fresh = {}
            for row in session.run(_EDGES, name=name, type=node_type):
                key = (row["source"], row["source_type"], row["relation"], row["target"], row["target_type"])
                if key not in seen and key not in fresh:
                    fresh[key] = row
            kept = sorted(fresh.values(), key=lambda row: len(row["chunk_ids"]), reverse=True)
            if hop > 1:
                kept = kept[:cap]
            for row in kept:
                seen.add((row["source"], row["source_type"], row["relation"], row["target"], row["target_type"]))
                edges.append({
                    "relation": row["relation"],
                    "source": row["source"], "source_type": row["source_type"],
                    "target": row["target"], "target_type": row["target_type"],
                    "descriptions": list(row["descriptions"]),
                    "chunk_ids": list(row["chunk_ids"]),
                    "hop": hop,
                })
                other = (row["other_name"], row["other_type"])
                if other not in visited:
                    visited.add(other)
                    next_frontier.add(other)
        frontier = next_frontier
    return edges


def rank_chunks(edges):
    """Order the provenance chunk_ids of traversed edges by relevance to the query.

    A chunk ranks higher the nearer to the seeds it was reached (shallowest hop) and the
    more edges in the neighborhood cite it (more connective). The chunk_id breaks ties so
    the order is stable.
    """
    nearest = {}
    frequency = {}
    for edge in edges:
        for chunk_id in edge["chunk_ids"]:
            nearest[chunk_id] = min(nearest.get(chunk_id, edge["hop"]), edge["hop"])
            frequency[chunk_id] = frequency.get(chunk_id, 0) + 1
    return sorted(nearest, key=lambda chunk_id: (nearest[chunk_id], -frequency[chunk_id], chunk_id))


def link_query(query, session):
    """Link the capitalized spans in the query to seed (name, type) nodes in the graph."""
    nodes = [(row["name"], row["type"]) for row in session.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type")]
    return link_spans(capitalized_spans(query), nodes)


def graph_chunk_ids(query, session, hops=2, cap=10, limit=30):
    """Chunk ids reached from the query's entities in the graph, ranked by relevance.

    Links the query to seed nodes, walks their SAME_AS-expanded neighborhood, and ranks the
    provenance chunk_ids so a graph traversal can act as a retriever alongside vector and BM25.
    """
    edges = neighborhood(link_query(query, session), session, hops=hops, cap=cap)
    return rank_chunks(edges)[:limit]
