from __future__ import annotations

from transformers import AutoTokenizer

from ..config import ChunkingConfig
from ..models import Chunk, Document
from .base import Chunker, make_chunk_id

class FixedSizeTokenChunker(Chunker):
    def __init__(self, cfg: ChunkingConfig, tokenizer=None):
        if not 0 <= cfg.chunk_overlap < cfg.chunk_size:
            raise ValueError("require 0 <= chunk_overlap < chunk_size")
        self.cfg = cfg
        self._tokenizer = tokenizer 

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name, use_fast=True)
        return self._tokenizer

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []        
        enc = self.tokenizer(text, add_special_tokens=False,
                             return_offsets_mapping=True, truncation=False,
                             verbose=False)
        # return_offsets_mapping=True return pairs [start_pos_token, end__pos_token)
        
        offsets = enc["offset_mapping"]
        n = len(offsets)
        size, stride = self.cfg.chunk_size, self.cfg.stride
        out: list[Chunk] = []
        for idx, start in enumerate(range(0, n, stride)):
            end = min(start + size, n)
            cs, ce = offsets[start][0], offsets[end - 1][1]
            out.append(Chunk(
                chunk_id=make_chunk_id(doc.document_id, idx),
                document_id=doc.document_id, chunk_index=idx,
                chunk_text=text[cs:ce], title=doc.title,
                source_url=doc.source_url, domain=doc.domain,
                language=doc.language, token_count=end - start,
                char_start=cs, char_end=ce))
            if end >= n:
                break
        return out