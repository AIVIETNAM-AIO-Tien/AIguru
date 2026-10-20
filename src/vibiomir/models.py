from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Document:
    document_id: str           
    source_url: str
    domain: str
    language: str               # "vi" | "en" | "zh"
    title: Optional[str]
    text: str                  
    provenance: dict[str, Any] = field(default_factory=dict)  


@dataclass(frozen=True)
class Chunk:
    chunk_id: str               # f"{document_id}::{chunk_index}" (chunking.base.make_chunk_id)
    document_id: str
    chunk_index: int
    chunk_text: str             # original text
    title: Optional[str]
    source_url: str
    domain: str
    language: str
    token_count: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    tokenized_text: Optional[str] = None   # optional lexical field (pyvi output)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    query_text: str


@dataclass
class Candidate:
    """One ranked hit from a single retriever."""
    chunk_id: str
    document_id: str
    rank: int                   # 1-based 
    score: float                # raw score; interpretation is retriever-specific
    source: str                 # "dense" | "bm25" | "rrf"
    chunk_text: Optional[str] = None
    source_url: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedCandidate(Candidate):
    """Candidate after fusion; keeps per-source ranks for analysis."""
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RankedDocument:
    document_id: str
    rank: int
    score: float
    source_url: Optional[str] = None
    supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: Query
    chunks: list[FusedCandidate]          # full ranked chunk list, preserved
    documents: list[RankedDocument]       # unique document_ids
