# ABOUTME: Graph-aware retrieval — links a question to graph nodes and gathers evidence from their neighborhood.
# ABOUTME: An LLM names the question's entities; their SAME_AS-expanded neighborhood yields candidate bridge chunks.

import collections

from graphrag_wiki import llm
from graphrag_wiki.config import CORPUS_PATH
from graphrag_wiki.corpus import iter_chunks, load_corpus
from graphrag_wiki.entity_resolution import normalize_name
from graphrag_wiki.graph_schema import NODE_TYPES
from graphrag_wiki.vector import rank_by_query

ENTITY_FORMAT = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}

ENTITY_PROMPT_TEMPLATE = (
    "List the entities this question about ancient Rome is asking about, so they can be "
    "looked up by name in a knowledge graph. Cover people, places, organizations, events, "
    "laws, and concepts ({types}) — both those named outright and those the question clearly "
    "implies. Give each as its shortest standalone name.\n\n"
    "Question: {query}"
)


def build_entity_prompt(query):
    """Prompt asking the model to name the entities and concepts a question is about."""
    return ENTITY_PROMPT_TEMPLATE.format(query=query, types=", ".join(NODE_TYPES).lower())


def extract_query_entities(query):
    """Ask the model which entities and concepts the question is about."""
    return llm.generate(build_entity_prompt(query), ENTITY_FORMAT, "graph_retrieval")["entities"]


def link_spans(spans, nodes):
    """Return the (name, type) nodes whose normalized name equals one of the spans.

    nodes is a list of (name, type). Matching is on the normalized name, so case and
    punctuation are ignored but distinct names stay distinct — "Roman emperor" does not
    collapse onto "Roman king". A name shared by nodes of different types links all of
    them; spans matching nothing are dropped.
    """
    index = collections.defaultdict(list)
    for name, node_type in nodes:
        index[normalize_name(name)].append((name, node_type))

    matched = set()
    for span in spans:
        matched.update(index.get(normalize_name(span), []))
    return matched


def seeds_from_entities(entities, session):
    """Match the extracted entity names to seed (name, type) nodes in the graph."""
    nodes = [(row["name"], row["type"]) for row in session.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type")]
    return link_spans(entities, nodes)


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


def candidate_chunk_ids(edges):
    """The distinct provenance chunk_ids of the traversed edges, in first-seen order."""
    ordered = []
    present = set()
    for edge in edges:
        for chunk_id in edge["chunk_ids"]:
            if chunk_id not in present:
                present.add(chunk_id)
                ordered.append(chunk_id)
    return ordered


def graph_passages(query, session, hops=2, cap=10, top_m=6):
    """Return the top-m chunk records reached from the query's entities through the graph.

    Names the question's entities with the model, links them to seed nodes, walks their
    SAME_AS-expanded neighborhood for candidate provenance chunks, and reranks those by
    relevance to the question. These bridge chunks stand apart from lexical/dense retrieval.
    """
    seeds = seeds_from_entities(extract_query_entities(query), session)
    edges = neighborhood(list(seeds), session, hops=hops, cap=cap)
    corpus = {chunk["chunk_id"]: chunk for chunk in iter_chunks(load_corpus(CORPUS_PATH))}
    candidates = [corpus[chunk_id] for chunk_id in candidate_chunk_ids(edges) if chunk_id in corpus]
    return rank_by_query(query, candidates)[:top_m]
