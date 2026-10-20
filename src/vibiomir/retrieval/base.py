"""Retriever interface (open to future retrievers, e.g. sparse/multi-vector/reranker)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Candidate


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query_text: str, k: int) -> list[Candidate]:
        """Return up to k candidates, best first, with rank starting at 1."""
