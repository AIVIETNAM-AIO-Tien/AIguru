from __future__ import annotations

from ..models import Candidate, FusedCandidate


class RRFFusion:
    """RRF_score(d) = sum_i 1 / (k + rank_i(d)), ranks 1-based, k default 60.

    Contract: dedupe by chunk_id across lists; a chunk absent from a list contributes nothing for
    that list; ties broken deterministically (TODO: define, e.g. by best source rank then chunk_id);
    FusedCandidate keeps per-source ranks/scores.
    """

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def fuse(self, ranked_lists: dict[str, list[Candidate]]) -> list[FusedCandidate]:
        raise NotImplementedError("TODO: RRF fusion")
