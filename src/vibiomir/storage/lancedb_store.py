from __future__ import annotations

import logging
import lancedb 

import pyarrow as pa
import numpy as np

from typing import Any, Iterable, Optional, Sequence

from ..config import LanceDBConfig
from ..models import Chunk

log = logging.getLogger(__name__)

def chunk_schema(vector_dim: int, with_tokenized_text: bool = True):
    """Explicit pyarrow schema; vector dimension comes from the embedding model, not a constant."""
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
        fields.insert(8, pa.field("tokenized_text", pa.string()))  # pyvi
    return pa.schema(fields)

def sql_quote(value: str) -> str:
    """Đặt chuỗi vào điều kiện SQL của LanceDB: gấp đôi dấu nháy đơn."""
    return "'" + str(value).replace("'", "''") + "'"

class LanceDBClient:
    def __init__(self, cfg: LanceDBConfig):
        self.cfg = cfg
        self._db = None
        
    @property
    def db(self):
        if self._db is None:
            self.connect()
        return self._db

    def connect(self):
        self._db = lancedb.connect(self.cfg.uri)
        return self._db
        
    def table_exists(self) -> bool:
        return self.cfg.table_name in self.db.table_names()

    def open_table(self):
        if not self.table_exists():
            raise FileNotFoundError(f"{self.cfg.table_name} doesnt exist in {self.cfg.uri}")
        return self.db.open_table(self.cfg.table_name)

    def create_table(self, vector_dim: int, *, overwrite: bool = False):
        if self.table_exists() and not overwrite:
            raise FileExistsError(f"{self.cfg.table_name} existed and overwrite = false")
        schema = chunk_schema(vector_dim, self.cfg.store_tokenized_text)
        return self.db.create_table(self.cfg.table_name, schema=schema, mode="overwrite" if overwrite else "create")


class ChunkRepository:
    """Storage only"""
    def __init__(self, client: LanceDBClient):
        self.client = client

    def add(self, chunks: Iterable[Chunk], vectors, *, embedding_model: str,
            embedding_version: str) -> int:
        """Append rows. Duplicate policy: caller must use `upsert` for re-runs."""
        table = self.client.open_table()
        arrow = self._to_arrow(table, chunks, vectors, embedding_model, embedding_version)
        
        if arrow is None:
            return 0
        table.add(arrow)
        return arrow.num_rows
    
    def upsert(self, chunks: Iterable[Chunk], vectors, *, embedding_model: str,
               embedding_version: str) -> int:
        """Merge on chunk_id (LanceDB merge_insert)."""
        table = self.client.open_table()
        arrow = self._to_arrow(table, chunks, vectors, embedding_model, embedding_version)
        if arrow is None:
            return 0
        result = (table.merge_insert("chunk_id")
                  .when_matched_update_all()
                  .when_not_matched_insert_all()
                  .execute(arrow))
        return int(result.num_updated_rows + result.num_inserted_rows)

    def get_by_ids(self, chunk_ids: Sequence[str], *, include_vector: bool = False) -> list[dict]:
        if not chunk_ids:
            return []
        table = self.client.open_table()
        columns = [n for n in table.schema.names if include_vector or n != "vector"]
        out: list[dict] = []
        for i in range(0, len(chunk_ids), 500):         
            batch = list(chunk_ids[i:i + 500])
            cond = "chunk_id IN (" + ", ".join(sql_quote(c) for c in batch) + ")"
            out += table.search().where(cond).select(columns).limit(len(batch)).to_list()
        return out

    def count(self) -> int:
        return self.client.open_table().count_rows()

    def delete_document(self, document_id: str) -> None:
        self.client.open_table().delete(f"document_id = {sql_quote(document_id)}")
    
    @staticmethod
    def _to_arrow(table, chunks: Sequence[Chunk], vectors: np.ndarray, model: str, version: str):
        """convert (chunks, vectors) -> arrows schema
        """
        if len(chunks) == 0:
            return None
        dim = table.schema.field("vector").type.list_size
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), dim):
            raise ValueError(f"vector shape: {vectors.shape}, expected {(len(chunks),dim)}")
        
        if not np.isfinite(vectors).all():
            raise ValueError(f"Vectors have NaN or inf")
        
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id must not be duplicate")

        data: dict[str, Any] = {
            "chunk_id": ids,
            "document_id": [c.document_id for c in chunks],
            "chunk_index": [c.chunk_index for c in chunks],
            "chunk_text": [c.chunk_text for c in chunks],
            "title": [c.title for c in chunks],
            "source_url": [c.source_url for c in chunks],
            "domain": [c.domain for c in chunks],
            "language": [c.language for c in chunks],
            "token_count": [c.token_count for c in chunks],
            "embedding_model": [model] * len(chunks),
            "embedding_version": [version] * len(chunks),
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), dim),
        }
        if "tokenized_text" in table.schema.names:
            data["tokenized_text"] = [c.tokenized_text for c in chunks]
        return pa.Table.from_pydict(data, schema=table.schema)