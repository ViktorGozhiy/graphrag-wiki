# ABOUTME: Entity resolution — finds nodes that name the same real-world thing across chunks.
# ABOUTME: Cheap blocking (shared name tokens) narrows the pairs an LLM then adjudicates into SAME_AS links.

import collections
import re

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
