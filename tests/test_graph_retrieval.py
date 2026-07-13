# ABOUTME: Tests for graph-aware retrieval — linking a question to graph nodes and traversing the neighborhood.
# ABOUTME: Covers capitalized-span extraction, name-index matching, and SAME_AS-expanded traversal.

from graphrag_wiki.graph_retrieval import capitalized_spans, link_spans


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
