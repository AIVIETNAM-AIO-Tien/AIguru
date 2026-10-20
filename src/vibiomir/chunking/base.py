"""Chunker interface and stable chunk-id helper."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from ..models import Chunk, Document


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Stable, reproducible id. Separator '::' assumed absent from benchmark ids (UNVERIFIED)."""
    return f"{document_id}::{chunk_index}"


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one document. Must not truncate text; must keep document_id/url/domain."""

    def chunk_many(self, docs: Iterable[Document]) -> Iterator[Chunk]:
        for d in docs:
            yield from self.chunk(d)
