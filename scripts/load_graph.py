# ABOUTME: Loads the extracted knowledge graph into Neo4j — entities become nodes, relations become edges.
# ABOUTME: Re-running is idempotent (MERGE on name+type), accumulating provenance rather than duplicating.

import json

from graphrag_wiki import graph_store

EXTRACTIONS_PATH = "data/graph/extractions.jsonl"


def iter_records(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main():
    records = list(iter_records(EXTRACTIONS_PATH))
    print(f"records: {len(records)} — loading into Neo4j ...")

    driver = graph_store.driver()
    driver.verify_connectivity()
    with driver.session() as session:
        counts = graph_store.load_records(records, session)
        nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edges = session.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c").single()["c"]
    driver.close()

    print(f"loaded: {counts}")
    print(f"graph now: {nodes} nodes, {edges} edges")


if __name__ == "__main__":
    main()
