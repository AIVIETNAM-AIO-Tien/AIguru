"""LanceDB connection, schema and chunk repository (interfaces only).

Version note: LanceDB's FTS / vector-index APIs differ across releases (native FTS vs
tantivy-based; create_fts_index / create_index signatures). TODO: pin the version and
verify against `lancedb.__version__` before implementing.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..config import LanceDBConfig
from ..models import Chunk


def chunk_schema(vector_dim: int, with_tokenized_text: bool = True):
    """Explicit pyarrow schema; vector dimension comes from the embedding model, not a constant."""
    import pyarrow as pa
    fields = [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("chunk_text", pa.string(), nullable=False),     # original, unmodified
        pa.field("title", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("language", pa.string()),
        pa.field("token_count", pa.int32()),
        pa.field("embedding_model", pa.string()),
        pa.field("embedding_version", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
    ]
    if with_tokenized_text:
        fields.insert(8, pa.field("tokenized_text", pa.string()))  # optional lexical field (pyvi)
    return pa.schema(fields)


class LanceDBClient:
    """Connection manager. Lazy: importing lancedb and connecting happen on first use."""

    def __init__(self, cfg: LanceDBConfig):
        self.cfg = cfg
        self._db = None

    def connect(self):
        raise NotImplementedError("TODO: lancedb.connect(cfg.uri)")

    def open_table(self):
        raise NotImplementedError("TODO: open existing table")

    def create_table(self, vector_dim: int, *, overwrite: bool = False):
        raise NotImplementedError("TODO: create table from chunk_schema(vector_dim)")


class ChunkRepository:
    """Storage only: no embedding or retrieval logic. Vectors are passed in by the caller."""

    def __init__(self, client: LanceDBClient):
        self.client = client

    def add(self, chunks: Iterable[Chunk], vectors, *, embedding_model: str,
            embedding_version: str) -> int:
        """Append rows. Duplicate policy: caller must use `upsert` for re-runs."""
        raise NotImplementedError("TODO")

    def upsert(self, chunks: Iterable[Chunk], vectors, *, embedding_model: str,
               embedding_version: str) -> int:
        """Merge on chunk_id (LanceDB merge_insert; verify availability in pinned version)."""
        raise NotImplementedError("TODO")

    def get_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        raise NotImplementedError("TODO")

    def count(self) -> int:
        raise NotImplementedError("TODO")

    def delete_document(self, document_id: str) -> int:
        raise NotImplementedError("TODO")
