# ABOUTME: Tests for entity-resolution blocking — the cheap candidate-pair generation before the LLM decides.
# ABOUTME: Covers name normalization, honorific-stripped tokens, and token blocking with a block-size cap.

from graphrag_wiki.entity_resolution import (
    DECISION_FORMAT,
    build_decision_prompt,
    candidate_pairs,
    embedding_pairs,
    name_tokens,
    normalize_name,
    token_block_pairs,
)


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


def test_embedding_pairs_link_close_vectors_within_type():
    vectors = [[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]]
    types = ["Concept", "Concept", "Concept"]
    assert embedding_pairs(vectors, types, theta=0.9) == {(0, 1)}


def test_embedding_pairs_never_cross_types():
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    types = ["Concept", "Person"]
    assert embedding_pairs(vectors, types, theta=0.9) == set()


def test_candidate_pairs_union_filtered_to_hub_endpoints():
    name_pairs = {(0, 1)}
    emb_pairs = {(1, 2)}
    # hub = node 0 only: (0,1) touches a hub and stays; (1,2) touches no hub and drops.
    assert candidate_pairs(name_pairs, emb_pairs, hubs={0}) == {(0, 1)}
    # hub = node 1: both pairs touch it.
    assert candidate_pairs(name_pairs, emb_pairs, hubs={1}) == {(0, 1), (1, 2)}


def test_decision_format_is_strict():
    assert DECISION_FORMAT["additionalProperties"] is False
    assert set(DECISION_FORMAT["required"]) == {"same", "canonical_name"}


def test_decision_prompt_includes_names_type_and_descriptions():
    a = {"name": "Caracalla", "type": "Person", "descriptions": ["Roman emperor.", "Son of Severus."]}
    b = {"name": "Emperor Caracalla", "type": "Person", "descriptions": ["Issued the edict of 212."]}
    prompt = build_decision_prompt(a, b)
    assert "Caracalla" in prompt and "Emperor Caracalla" in prompt
    assert "Person" in prompt
    assert "Roman emperor." in prompt and "Issued the edict of 212." in prompt
    assert "canonical_name" in prompt


def test_decision_prompt_caps_descriptions():
    a = {"name": "X", "type": "Concept", "descriptions": [f"desc{i}" for i in range(10)]}
    b = {"name": "Y", "type": "Concept", "descriptions": ["z"]}
    prompt = build_decision_prompt(a, b)
    assert "desc0" in prompt and "desc2" in prompt
    assert "desc3" not in prompt  # capped at MAX_DESCRIPTIONS
