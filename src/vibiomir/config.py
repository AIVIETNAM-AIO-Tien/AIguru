from __future__ import annotations

import yaml
import logging

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PathsConfig:
    profiling_csv: str = "data/profing.csv"
    parquet_dir: str = "data/corpus"
    output_dir: str = "outputs"


@dataclass
class ProfilingConfig:
    encoding: str = "utf-8-sig"
    columns: dict[str, str] = field(default_factory=lambda: {
        "domain": "ten_mien", "language": "ngon_ngu", "url_count": "so_luong_url"})

@dataclass
class CorpusConfig:
    filename_pattern: str = "{domain}.parquet"
    columns: dict[str, str] = field(default_factory=lambda: {
        k: k for k in ("id", "url", "domain", "title", "text")})
    languages: list[str] = field(default_factory=lambda: ["vi"])


@dataclass
class ErrorPolicyConfig:
    on_missing_file: str = "warn"
    on_row_count_mismatch: str = "warn"
    on_read_error: str = "warn"
    on_empty_text: str = "keep"


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 100
    tokenizer_name: str = "BAAI/bge-m3"

    @property
    def stride(self) -> int:
        return self.chunk_size - self.chunk_overlap


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    revision: Optional[str] = None
    batch_size: int = 16
    normalize: bool = True
    max_length: int = 512


@dataclass
class LanceDBConfig:
    uri: str = "lancedb_data"
    table_name: str = "chunks"
    metric: str = "cosine"
    store_tokenized_text: bool = True


@dataclass
class RetrievalConfig:
    dense_k: int = 100
    bm25_k: int = 100
    rrf_k: int = 60
    final_chunk_k: int = 20
    final_doc_k: int = 20
    doc_policy: str = "best_chunk"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    errors: ErrorPolicyConfig = field(default_factory=ErrorPolicyConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    lancedb: LanceDBConfig = field(default_factory=LanceDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge(obj: Any, data: dict[str, Any]) -> None:
    unknown = set(data) - {f.name for f in fields(obj)}
    if unknown:
        raise KeyError(f"Unknown config keys for {type(obj).__name__}: {sorted(unknown)}")
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        if is_dataclass(cur) and isinstance(data[f.name], dict):
            _merge(cur, data[f.name])
        else:
            setattr(obj, f.name, data[f.name])


def load_settings(path: str | Path | None = None) -> Settings:
    s = Settings()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            _merge(s, yaml.safe_load(f) or {})
    return s

def setup_logging(cfg: LoggingConfig) -> None:
    logging.basicConfig(level=cfg.level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
