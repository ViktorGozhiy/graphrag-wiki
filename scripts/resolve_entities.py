# ABOUTME: Resolves duplicate entities — blocks candidate pairs, asks the LLM, links confirmed ones.
# ABOUTME: Resumable (skips pairs already decided) and concurrent; then writes SAME_AS edges in Neo4j.

import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastembed import TextEmbedding

from graphrag_wiki import graph_store
from graphrag_wiki.config import EMBED_MODEL
from graphrag_wiki.entity_resolution import (
    candidate_pairs,
    decide_same,
    embedding_pairs,
    token_block_pairs,
)

EXTRACTIONS_PATH = "data/graph/extractions.jsonl"
DECISIONS_PATH = "data/graph/resolutions.jsonl"

THETA = 0.90          # embedding-blocking cosine floor
MAX_BLOCK = 5         # name-token blocks larger than this come from generic words, so drop them
HUB_DEGREE = 2        # only resolve pairs touching a node with at least this many edges
MAX_WORKERS = 12
FAILURE_ABORT = 20    # stop if the endpoint is clearly dead


def load_nodes():
    """Distinct (name, type) nodes with the descriptions the extractor gave each."""
    node_desc = collections.defaultdict(list)
    with open(EXTRACTIONS_PATH, encoding="utf-8") as handle:
        for line in handle:
            for entity in json.loads(line)["entities"]:
                node_desc[(entity["name"], entity["type"])].append(entity.get("description", ""))
    return list(node_desc), node_desc


def node_degrees():
    """Post-merge degree of every node, read from the loaded graph."""
    degree = {}
    driver = graph_store.driver()
    with driver.session() as session:
        for r in session.run(
            "MATCH (n:Entity) RETURN n.name AS name, n.type AS type, COUNT{(n)--()} AS deg"
        ):
            degree[(r["name"], r["type"])] = r["deg"]
    driver.close()
    return degree


def node_dict(key, node_desc):
    return {"name": key[0], "type": key[1], "descriptions": node_desc[key]}


def pair_identity(a, b):
    """Order-independent identity for a pair, as sorted [[name, type], [name, type]]."""
    return sorted([[a[0], a[1]], [b[0], b[1]]])


def decided_pairs(path):
    """Pair identities already decided, so a resumed run skips them."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.dumps(json.loads(line)["pair"]) for line in handle if line.strip()}


def link_confirmed():
    """Write a SAME_AS edge for every pair the model confirmed as the same entity."""
    links = []
    with open(DECISIONS_PATH, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["same"]:
                (a_name, a_type), (b_name, b_type) = record["pair"]
                links.append(
                    {
                        "a": {"name": a_name, "type": a_type},
                        "b": {"name": b_name, "type": b_type},
                        "canonical": record["canonical"],
                    }
                )
    driver = graph_store.driver()
    with driver.session() as session:
        graph_store.link_same_as(links, session)
    driver.close()
    print(f"linked {len(links)} SAME_AS edges", flush=True)


def main():
    os.makedirs(os.path.dirname(DECISIONS_PATH), exist_ok=True)
    nodes, node_desc = load_nodes()
    types = [node_type for _, node_type in nodes]
    degree = node_degrees()
    hubs = {i for i, key in enumerate(nodes) if degree.get(key, 0) >= HUB_DEGREE}

    print(f"nodes: {len(nodes)}  hubs(deg>={HUB_DEGREE}): {len(hubs)} — embedding ...", flush=True)
    reps = [
        f"{name} — {max(node_desc[(name, t)], key=len)}" if node_desc[(name, t)] else name
        for name, t in nodes
    ]
    vectors = np.array(list(TextEmbedding(model_name=EMBED_MODEL).embed(reps)), dtype=np.float32)

    name_pairs = token_block_pairs(nodes, MAX_BLOCK)
    emb_pairs = embedding_pairs(vectors, types, THETA)
    candidates = candidate_pairs(name_pairs, emb_pairs, hubs)
    print(f"candidates: {len(candidates)} (name {len(name_pairs)}, embed {len(emb_pairs)})", flush=True)

    done = decided_pairs(DECISIONS_PATH)
    pending = [(i, j) for i, j in candidates if json.dumps(pair_identity(nodes[i], nodes[j])) not in done]
    print(f"pending decisions: {len(pending)}  (already done: {len(done)})", flush=True)

    successes = 0
    failures = 0
    confirmed = 0
    with open(DECISIONS_PATH, "a", encoding="utf-8") as out, ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {
            pool.submit(decide_same, node_dict(nodes[i], node_desc), node_dict(nodes[j], node_desc)): (i, j)
            for i, j in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            i, j = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures += 1
                print(
                    f"[{index}/{len(pending)}] {nodes[i][0]!r}~{nodes[j][0]!r} FAILED "
                    f"({type(error).__name__}: {error}) — will retry on next run",
                    flush=True,
                )
                if successes == 0 and failures >= FAILURE_ABORT:
                    print(f"{failures} failures and nothing decided — endpoint down; stopping", flush=True)
                    for other in futures:
                        other.cancel()
                    break
                continue
            successes += 1
            out.write(
                json.dumps(
                    {
                        "pair": pair_identity(nodes[i], nodes[j]),
                        "same": result["same"],
                        "canonical": result["canonical_name"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            if result["same"]:
                confirmed += 1
                print(
                    f"[{index}/{len(pending)}] SAME {nodes[i][0]!r} ~ {nodes[j][0]!r} "
                    f"-> {result['canonical_name']!r}",
                    flush=True,
                )

    print(f"decided {successes}, confirmed same: {confirmed}", flush=True)
    link_confirmed()


if __name__ == "__main__":
    main()
