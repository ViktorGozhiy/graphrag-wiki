# ABOUTME: Reciprocal Rank Fusion — merges ranked lists by rank position, not by raw score.
# ABOUTME: Rewards chunks that several retrievers agree on, sidestepping incompatible score scales.

def reciprocal_rank_fusion(ranked_lists, k=60, weights=None):
    """Fuse ranked lists of chunk_ids into one ranking.

    Each list is ordered best-first. A chunk's score is the weighted sum of
    1 / (k + rank) over the lists it appears in. Returns (chunk_id, score)
    pairs sorted by score descending, ties broken by chunk_id ascending.
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
