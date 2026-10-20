from __future__ import annotations

from ..chunking.base import Chunker
from ..config import Settings
from ..data.parquet_reader import ParquetCorpusReader
from ..data.profiling import DomainInventory
from ..embeddings.base import EmbeddingModel
from ..storage.index_manager import IndexManager
from ..storage.lancedb_store import ChunkRepository


class IndexingPipeline:
    """Flow: reader -> chunker -> embedder -> repository (ingest) -> IndexManager (indexes).
    Ingestion and index creation are separate steps."""

    def __init__(self, settings: Settings, reader: ParquetCorpusReader, chunker: Chunker,
                 embedder: EmbeddingModel, repo: ChunkRepository, indexes: IndexManager):
        self.s, self.reader, self.chunker = settings, reader, chunker
        self.embedder, self.repo, self.indexes = embedder, repo, indexes

    def ingest(self, inventory: DomainInventory) -> None:
        raise NotImplementedError("TODO: stream domains in batches, upsert chunks + vectors")

    def build_indexes(self) -> None:
        raise NotImplementedError("TODO: vector + FTS index creation")
