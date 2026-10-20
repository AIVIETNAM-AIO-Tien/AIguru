from __future__ import annotations

from ..config import RetrievalConfig
from ..models import FusedCandidate
from .base import Retriever
from .fusion import RRFFusion


class HybridRetriever:
    """Runs N retrievers (dense + bm25 now), fuses with RRF. Additional retrievers can be added
    to `retrievers` without changing this class."""

    def __init__(self, retrievers: dict[Retriever, int] | list[tuple[Retriever, int]],
                 fusion: RRFFusion):
        self.retrievers = list(retrievers.items()) if isinstance(retrievers, dict) else list(retrievers)
        self.fusion = fusion

    def retrieve(self, query_text: str) -> list[FusedCandidate]:
        """TODO: call each (retriever, k), fuse; do NOT truncate here (final K is applied later)."""
        raise NotImplementedError("TODO: hybrid retrieval")
