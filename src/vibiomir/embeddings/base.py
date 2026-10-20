"""Embedding model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class EmbeddingModel(ABC):
    """Contract: both methods return float32 ndarray of shape (n, dimension),
    row-aligned with the input order. L2-normalized iff `normalized` is True."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def revision(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def normalized(self) -> bool: ...

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Returns shape (dimension,)."""

    def describe(self) -> dict:
        """Recorded next to the index (embedding_model / embedding_version)."""
        return {"model_name": self.model_name, "revision": self.revision,
                "dimension": self.dimension, "normalized": self.normalized}
