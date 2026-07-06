# ABOUTME: Tests for Reciprocal Rank Fusion of ranked chunk_id lists.
# ABOUTME: Covers consensus reward, deterministic tie-breaking, and weighting.

from graphrag_wiki.fusion import reciprocal_rank_fusion


def test_consensus_beats_single_list_and_is_deterministic():
    # A (v#1, b#3) and C (v#3, b#1) appear in both lists; B and D appear once.
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["C", "D", "A"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["A", "C", "B", "D"]


def test_disjoint_lists_interleave_with_ties_broken_by_id():
    # No overlap: each rank level ties, broken by chunk_id ascending.
    fused = reciprocal_rank_fusion([["A", "B"], ["D", "E"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["A", "D", "B", "E"]


def test_weights_break_symmetry_in_disjoint_case():
    fused = reciprocal_rank_fusion([["A"], ["D"]], k=60, weights=[1.0, 0.5])
    scores = dict(fused)
    assert scores["A"] > scores["D"]
    assert [chunk_id for chunk_id, _ in fused] == ["A", "D"]
