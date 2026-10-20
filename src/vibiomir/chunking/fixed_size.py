"""Fixed-size, token-measured chunking with overlap (interface only)."""
from __future__ import annotations

from ..config import ChunkingConfig
from ..models import Chunk, Document
from .base import Chunker


class FixedSizeTokenChunker(Chunker):
    """chunk_size/overlap are in TOKENS of `cfg.tokenizer_name` (default bge-m3's tokenizer).

    Candidate backends (decide in the implementation phase):
    - langchain-text-splitters `RecursiveCharacterTextSplitter.from_huggingface_tokenizer(...)`:
      length is measured with the tokenizer, but separators can yield chunks that are
      shorter than chunk_size and, rarely, boundary effects. Verify with a test on real text.
    - Direct tokenizer windowing with offset mappings (exact size/stride, char offsets for free).
    """

    def __init__(self, cfg: ChunkingConfig):
        if not 0 <= cfg.chunk_overlap < cfg.chunk_size:
            raise ValueError("require 0 <= chunk_overlap < chunk_size")
        self.cfg = cfg
        self._tokenizer = None  # lazy

    def chunk(self, doc: Document) -> list[Chunk]:
        # TODO: load tokenizer lazily, split by tokens with overlap, fill token_count and
        # char offsets, build ids via make_chunk_id, and never drop text.
        raise NotImplementedError("TODO: fixed-size token chunking")
