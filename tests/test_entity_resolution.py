# ABOUTME: Tests for entity-resolution blocking — the cheap candidate-pair generation before the LLM decides.
# ABOUTME: Covers name normalization, honorific-stripped tokens, and token blocking with a block-size cap.

from graphrag_wiki.entity_resolution import name_tokens, normalize_name, token_block_pairs


def test_normalize_lowercases_and_drops_punctuation():
    assert normalize_name("Octavian (Augustus)") == "octavian augustus"
    assert normalize_name("  Emperor   Caracalla ") == "emperor caracalla"


def test_name_tokens_strip_honorifics():
    assert name_tokens("Emperor Caracalla") == frozenset({"caracalla"})
    assert name_tokens("Caracalla") == frozenset({"caracalla"})
    assert name_tokens("Octavian (Augustus)") == frozenset({"octavian", "augustus"})


def test_token_block_pairs_links_shared_token_same_type():
    nodes = [
        ("Caracalla", "Person"),
        ("Emperor Caracalla", "Person"),
        ("citizenship", "Concept"),
    ]
    assert token_block_pairs(nodes, max_block=10) == {(0, 1)}


def test_token_block_pairs_never_crosses_types():
    nodes = [("Rome", "Place"), ("Rome", "Concept")]
    assert token_block_pairs(nodes, max_block=10) == set()


def test_token_block_pairs_discards_oversized_generic_blocks():
    # Four "Roman X" of one type share the generic token "roman"; with max_block=3 that
    # block is dropped, so no pairs come from it — but "empire" (size 2) still links its pair.
    nodes = [
        ("Roman Empire", "Organization"),
        ("Roman Republic", "Organization"),
        ("Roman army", "Organization"),
        ("Roman Senate", "Organization"),
        ("Empire", "Organization"),
    ]
    pairs = token_block_pairs(nodes, max_block=3)
    assert (0, 4) in pairs  # "Roman Empire" <-> "Empire" via the small "empire" block
    # no pair is created solely through the oversized "roman" block
    assert (1, 2) not in pairs and (2, 3) not in pairs and (1, 3) not in pairs
