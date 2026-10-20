from __future__ import annotations

from typing import Callable, Optional

from ..config import RetrievalConfig
from ..models import Query, RetrievalResult
from ..retrieval.document_ranker import DocumentRanker
from ..retrieval.hybrid import HybridRetriever


class QueryPipeline:
    """normalize (optional) -> hybrid retrieve -> truncate to final_chunk_k -> rank documents
    (-> truncate to final_doc_k). Serialization lives in `submission/`, not here."""

    def __init__(self, hybrid: HybridRetriever, doc_ranker: DocumentRanker, cfg: RetrievalConfig,
                 normalizer: Optional[Callable[[str], str]] = None):
        self.hybrid, self.doc_ranker, self.cfg, self.normalizer = hybrid, doc_ranker, cfg, normalizer

    def run(self, query: Query) -> RetrievalResult:
        raise NotImplementedError("TODO: orchestrate query flow")
