"""BAAI/bge-m3 dense embedding (lazy; nothing is downloaded at import/instantiation)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import EmbeddingModel


class BgeM3EmbeddingModel(EmbeddingModel):
    """Dense only. Sparse / ColBERT outputs are deliberate extension points (not implemented).

    TODO: choose backend (transformers vs FlagEmbedding), pin revision, confirm dimension (expected
    1024, UNVERIFIED until the model is loaded) and whether query/document need prefixes (bge-m3: none).
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = None  # lazy

    model_name = property(lambda self: self.cfg.model_name)
    revision = property(lambda self: self.cfg.revision)
    normalized = property(lambda self: self.cfg.normalize)

    @property
    def dimension(self) -> int:
        # TODO: read from the loaded model config instead of a constant.
        raise NotImplementedError("TODO: dimension is known only after the model is loaded")

    def _load(self) -> None:
        raise NotImplementedError("TODO: lazy model loading")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("TODO: batched dense encoding")

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError("TODO: query encoding")
