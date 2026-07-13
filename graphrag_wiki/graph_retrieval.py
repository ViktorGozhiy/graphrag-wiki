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
