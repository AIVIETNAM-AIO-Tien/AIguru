"""Index lifecycle, kept separate from data ingestion."""
from __future__ import annotations

from ..config import LanceDBConfig
from .lancedb_store import LanceDBClient


class IndexManager:
    """Lifecycle: ingest rows -> create indexes -> serve. Indexes are rebuilt (not patched) when
    the embedding model/revision or vector dimension changes; the stored `embedding_model` /
    `embedding_version` columns are compared with the active model before querying.
    """

    def __init__(self, client: LanceDBClient, cfg: LanceDBConfig):
        self.client, self.cfg = client, cfg

    def create_vector_index(self, *, replace: bool = True) -> None:
        """TODO: exact search for pilot; benchmark IVF_PQ/HNSW later. Metric from cfg.metric."""
        raise NotImplementedError

    def create_fts_index(self, column: str = "chunk_text", *, replace: bool = True) -> None:
        """TODO: `column` is 'tokenized_text' when Vietnamese word segmentation is used;
        record tokenizer config next to the index."""
        raise NotImplementedError

    def needs_rebuild(self, model_name: str, revision: str | None, dimension: int) -> bool:
        """TODO: compare against metadata stored with the table."""
        raise NotImplementedError

    def drop_indexes(self) -> None:
        raise NotImplementedError
