from __future__ import annotations

from typing import Callable, Optional

from ..models import Candidate
from ..storage.lancedb_store import LanceDBClient
from .base import Retriever


class LanceDBBM25Retriever(Retriever):
    """LanceDB FTS branch. Independent from bge-m3 sparse weights.

    `query_tokenizer` is the extension point for Vietnamese segmentation (e.g. pyvi); it MUST be the
    same function used to produce the indexed `tokenized_text`. None = raw query on `chunk_text`.
    """
    name = "bm25"

    def __init__(self, client: LanceDBClient, text_column: str = "chunk_text",
                 query_tokenizer: Optional[Callable[[str], str]] = None):
        self.client, self.text_column, self.query_tokenizer = client, text_column, query_tokenizer

    def retrieve(self, query_text: str, k: int) -> list[Candidate]:
        raise NotImplementedError("TODO: FTS search (verify API in the pinned LanceDB version)")
