from __future__ import annotations

from ..embeddings.base import EmbeddingModel
from ..models import Candidate
from ..storage.lancedb_store import LanceDBClient
from .base import Retriever


class DenseRetriever(Retriever):
    """Encodes the query, runs vector search.

    Score contract: stores LanceDB's `_distance` in `raw`; `score` must be a
    higher-is-better value (e.g. 1 - cosine_distance) and the metric is cfg.lancedb.metric.
    TODO: confirm metric/normalization match the index.
    """
    name = "dense"

    def __init__(self, model: EmbeddingModel, client: LanceDBClient, metric: str = "cosine"):
        self.model, self.client, self.metric = model, client, metric

    def retrieve(self, query_text: str, k: int) -> list[Candidate]:
        raise NotImplementedError("TODO: dense search")
