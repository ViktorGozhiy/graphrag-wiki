# ABOUTME: Entity resolution — finds nodes that name the same real-world thing across chunks.
# ABOUTME: Cheap blocking (shared name tokens) narrows the pairs an LLM then adjudicates into SAME_AS links.

import collections
import re

import numpy as np

# Titles and particles that do not identify a specific entity, so two names that differ
# only by these should still block together ("Emperor Caracalla" ~ "Caracalla").
_NAME_STOPWORDS = frozenset(
    {"the", "of", "a", "an", "and", "emperor", "empress", "king", "queen",
     "saint", "st", "pope", "general", "consul", "prefect", "divus"}
)


def normalize_name(name):
    """Lowercase a name and reduce punctuation and runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name)).strip().lower()


def name_tokens(name):
    """Content tokens of a name — normalized words with honorifics and particles removed."""
    return frozenset(t for t in normalize_name(name).split() if t not in _NAME_STOPWORDS)


def token_block_pairs(nodes, max_block):
    """Candidate index pairs that share a name token within the same type.

    nodes is a list of (name, type). A shared token groups its nodes into a block; blocks
    larger than max_block come from generic tokens ("roman") that do not discriminate, so
    they are dropped. Returns a set of (i, j) index pairs with i < j.
    """
    blocks = collections.defaultdict(list)
    for index, (name, node_type) in enumerate(nodes):
        for token in name_tokens(name):
            blocks[(node_type, token)].append(index)

    pairs = set()
    for members in blocks.values():
        if 2 <= len(members) <= max_block:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
    return pairs


def embedding_pairs(vectors, types, theta):
    """Index pairs whose embeddings have cosine >= theta within the same type.

    vectors is one row per node (any scale — normalized here); types is the parallel
    type list. Returns a set of (i, j) index pairs with i < j.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    groups = collections.defaultdict(list)
    for index, node_type in enumerate(types):
        groups[node_type].append(index)

    pairs = set()
    for members in groups.values():
        idx = np.array(members)
        sim = vectors[idx] @ vectors[idx].T
        rows, cols = np.triu_indices(len(idx), k=1)
        for a, b in zip(idx[rows[sim[rows, cols] >= theta]], idx[cols[sim[rows, cols] >= theta]]):
            pairs.add((int(a), int(b)))
    return pairs


def candidate_pairs(name_pairs, emb_pairs, hubs):
    """Union the two blockers, keeping only pairs where at least one endpoint is a hub."""
    hubs = set(hubs)
    return {pair for pair in name_pairs | emb_pairs if pair[0] in hubs or pair[1] in hubs}
