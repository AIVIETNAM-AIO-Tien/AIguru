from __future__ import annotations

from ..models import FusedCandidate, RankedDocument


class DocumentRanker:
    """Chunk ranking -> document ranking. The chunk list is never modified.

    Policies: "best_chunk" (initial: doc score = score of its best-ranked chunk). Others
    (sum / top-n mean) can be added behind `policy`. Output has unique document_ids and
    keeps supporting_chunk_ids and source_url for provenance.
    """

    def __init__(self, policy: str = "best_chunk"):
        if policy not in {"best_chunk"}:
            raise ValueError(f"unsupported policy {policy!r}")
        self.policy = policy

    def rank(self, chunks: list[FusedCandidate]) -> list[RankedDocument]:
        raise NotImplementedError("TODO: aggregate chunks by document_id")
