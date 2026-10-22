"""offline pipeline.
    Parquet gốc --chunking--> chunks --embed (GPU)--> vectors  --\
                                   \\-tokenize (CPU)-> tokens --index--> LanceDB
`paths.work_dir`, một shard = một domain:
    chunks/<shard>.parquet     thông tin chunk (không có vector)
    vectors/<shard>.parquet    chunk_id, vector      (cùng thứ tự dòng với chunks)
    tokens/<shard>.parquet     chunk_id, tokenized_text (cùng thứ tự dòng với chunks)
    manifests/<stage>__<shard>.json   số dòng, thời gian, phiên bản model...

Rule:
- Ghi vào file .tmp rồi đổi tên: shard bị ngắt giữa chừng không bị coi là đã xong.
- Shard đã có file thì BỎ QUA (trừ khi force=True) -> chạy lại/resume.
- Hai nhánh embed và tokenize đọc cùng file `chunks`, không chunk lại riêng.
- Bước index kiểm tra chunk_id của ba file khớp từng dòng, lệch thì dừng.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..chunking.base import Chunker
from ..data.parquet_reader import ParquetCorpusReader
from ..data.profiling import DomainRecord
from ..models import Chunk, Document
from ..storage.lancedb_store import ChunkRepository

log = logging.getLogger(__name__)

CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),
    pa.field("document_id", pa.string(), nullable=False),
    pa.field("chunk_index", pa.int32(), nullable=False),
    pa.field("chunk_text", pa.string(), nullable=False),
    pa.field("title", pa.string()),
    pa.field("source_url", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("language", pa.string()),
    pa.field("token_count", pa.int32()),
    pa.field("char_start", pa.int32()),
    pa.field("char_end", pa.int32()),
])
TOKENS_SCHEMA = pa.schema([pa.field("chunk_id", pa.string(), nullable=False),
                           pa.field("tokenized_text", pa.string())])

def vectors_schema(dim: int) -> pa.Schema:
    return pa.schema([pa.field("chunk_id", pa.string(), nullable=False),
                      pa.field("vector", pa.list_(pa.float32(), dim))])

@dataclass
class StageStats:
    stage: str
    shard: str
    rows: int = 0
    seconds: float = 0.0
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_per_sec(self) -> float:
        return self.rows / self.seconds if self.seconds > 0 else 0.0

class WorkDir:
    def __init__(self, root: str | Path):
        self.root = Path(root)
    def path(self, kind: str, shard: str) -> Path:
        return self.root / kind / f"{shard}.parquet"
    def manifest(self, stage: str, shard: str) -> Path:
        return self.root / "manifests" / f"{stage}__{shard}.json"

@contextmanager
def atomic_parquet_writer(path: Path, schema: pa.Schema) -> Iterator[pq.ParquetWriter]:
    """Ghi vào .tmp, đóng xong mới đổi tên; lỗi thì xóa .tmp"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer = pq.ParquetWriter(tmp, schema)
    try:
        yield writer
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, path)

def write_manifest(work: WorkDir, stats: StageStats) -> None:
    p = work.manifest(stats.stage, stats.shard)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"stage": stats.stage, "shard": stats.shard, "rows": stats.rows,
            "seconds": round(stats.seconds, 3), "rows_per_sec": round(stats.rows_per_sec, 2),
            **stats.extra}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def iter_exact_batches(pf: pq.ParquetFile, columns: list[str], n: int) -> Iterator[pa.Table]:
    """Đọc file thành các bảng đúng n dòng, bất kể row group."""
    buf: Optional[pa.Table] = None
    for b in pf.iter_batches(batch_size=n, columns=columns):
        t = pa.Table.from_batches([b])
        buf = t if buf is None else pa.concat_tables([buf, t])
        while buf.num_rows >= n:
            yield buf.slice(0, n)
            buf = buf.slice(n)
    if buf is not None and buf.num_rows > 0:
        yield buf

def sample_documents(docs: Iterable[Document], n: int, seed: int) -> list[Document]:
    """Lấy ngẫu nhiên n tài liệu"""
    rng = random.Random(seed)
    kept: list[tuple[int, Document]] = []
    for i, d in enumerate(docs):
        if len(kept) < n:
            kept.append((i, d))
        else:
            j = rng.randint(0, i)
            if j < n:
                kept[j] = (i, d)
    return [d for _, d in sorted(kept, key=lambda x: x[0])]

# ----- chunk ------------------------------------------------------------------

def run_chunk(rec: DomainRecord, reader: ParquetCorpusReader, chunker: Chunker, work: WorkDir, *,
              batch_rows: int = 2048, max_docs: Optional[int] = None, seed: int = 0,
              force: bool = False) -> StageStats:
    shard, out = rec.domain, work.path("chunks", rec.domain)
    stats = StageStats("chunk", shard)
    if out.exists() and not force:
        stats.skipped = True
        return stats
    t0 = time.perf_counter()
    docs: Iterable[Document] = reader.iter_documents(rec)
    if max_docs is not None:
        docs = sample_documents(docs, max_docs, seed)
    n_docs = n_no_chunk = n_chars = 0
    rows: list[dict] = []

    def flush(writer):
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=CHUNKS_SCHEMA))
            rows.clear()

    with atomic_parquet_writer(out, CHUNKS_SCHEMA) as writer:
        for d in docs:
            n_docs += 1
            cs = chunker.chunk(d)
            if not cs:
                n_no_chunk += 1          
            for c in cs:
                stats.rows += 1
                n_chars += len(c.chunk_text)
                rows.append({k: getattr(c, k) for k in CHUNKS_SCHEMA.names})
            if len(rows) >= batch_rows:
                flush(writer)
        flush(writer)
    stats.seconds = time.perf_counter() - t0
    stats.extra = {"documents": n_docs, "documents_without_chunks": n_no_chunk,
                   "avg_chunk_chars": round(n_chars / stats.rows, 1) if stats.rows else 0,
                   "max_docs": max_docs, "seed": seed, "file_bytes": out.stat().st_size}
    write_manifest(work, stats)
    return stats

# ----- embed -----------------------------------------------------------------

def run_embed(shard: str, embedder, work: WorkDir, *, batch_rows: int = 2048,
              force: bool = False) -> StageStats:
    """`embedder` cần: encode_documents(list[str]) -> (n, dim) float32, `dimension`, `version_tag`."""
    src, out = work.path("chunks", shard), work.path("vectors", shard)
    stats = StageStats("embed", shard)
    if out.exists() and not force:
        stats.skipped = True
        return stats
    dim = embedder.dimension
    t0 = time.perf_counter()
    with atomic_parquet_writer(out, vectors_schema(dim)) as writer:
        for batch in iter_exact_batches(pq.ParquetFile(src), ["chunk_id", "chunk_text"], batch_rows):
            vecs = np.asarray(embedder.encode_documents(batch["chunk_text"].to_pylist()), dtype=np.float32)
            if vecs.shape != (batch.num_rows, dim):
                raise ValueError(f"{shard}: embedder trả {vecs.shape}, kỳ vọng {(batch.num_rows, dim)}")
            writer.write_table(pa.Table.from_pydict(
                {"chunk_id": batch["chunk_id"].to_pylist(),
                 "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), dim)},
                schema=vectors_schema(dim)))
            stats.rows += batch.num_rows
    stats.seconds = time.perf_counter() - t0
    stats.extra = {"version_tag": embedder.version_tag, "dimension": dim, "file_bytes": out.stat().st_size}
    write_manifest(work, stats)
    return stats

# ----- tokenize (pyvi) -------------------------------------------------------

def run_tokenize(shard: str, tokenize_fn: Callable[[str], str], work: WorkDir, *,
                 batch_rows: int = 2048, force: bool = False,
                 tokenizer_name: str = "pyvi") -> StageStats:
    src, out = work.path("chunks", shard), work.path("tokens", shard)
    stats = StageStats("tokenize", shard)
    if out.exists() and not force:
        stats.skipped = True
        return stats
    t0 = time.perf_counter()
    with atomic_parquet_writer(out, TOKENS_SCHEMA) as writer:
        for batch in iter_exact_batches(pq.ParquetFile(src), ["chunk_id", "chunk_text"], batch_rows):
            writer.write_table(pa.Table.from_pydict(
                {"chunk_id": batch["chunk_id"].to_pylist(),
                 "tokenized_text": [tokenize_fn(t) for t in batch["chunk_text"].to_pylist()]},
                schema=TOKENS_SCHEMA))
            stats.rows += batch.num_rows
    stats.seconds = time.perf_counter() - t0
    stats.extra = {"tokenizer": tokenizer_name, "file_bytes": out.stat().st_size}
    write_manifest(work, stats)
    return stats

# ----- index ------------------------------------------------

def run_index(shard: str, repo: ChunkRepository, work: WorkDir, *, embedding_model: str,
              embedding_version: str, batch_rows: int = 2048, mode: str = "upsert") -> StageStats:
    """Ghép chunks + vectors (+ tokens nếu bảng có cột tokenized_text) rồi nạp vào LanceDB.

    mode="upsert": an toàn khi chạy lại shard. mode="add": append, nhanh hơn nhưng chạy lại sẽ
    nhân đôi dòng; chỉ dùng cho bảng còn trống (chưa đo chênh lệch tốc độ trên dữ liệu lớn).
    """
    if mode not in {"upsert", "add"}:
        raise ValueError("mode phải là 'upsert' hoặc 'add'")
    write = repo.upsert if mode == "upsert" else repo.add
    paths = {"chunks": work.path("chunks", shard), "vectors": work.path("vectors", shard)}
    use_tokens = "tokenized_text" in repo.client.open_table().schema.names
    if use_tokens:
        paths["tokens"] = work.path("tokens", shard)
    for kind, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Thiếu file {kind} của shard {shard!r}: {p}")
    files = {k: pq.ParquetFile(p) for k, p in paths.items()}
    counts = {k: f.metadata.num_rows for k, f in files.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Shard {shard!r}: số dòng không khớp giữa các file: {counts}")

    t0 = time.perf_counter()
    stats = StageStats("index", shard)
    chunk_cols = CHUNKS_SCHEMA.names
    its = {"chunks": iter_exact_batches(files["chunks"], chunk_cols, batch_rows),
           "vectors": iter_exact_batches(files["vectors"], ["chunk_id", "vector"], batch_rows)}
    if use_tokens:
        its["tokens"] = iter_exact_batches(files["tokens"], ["chunk_id", "tokenized_text"], batch_rows)
    for parts in zip(*its.values()):
        by_kind = dict(zip(its.keys(), parts))
        ids = by_kind["chunks"]["chunk_id"].to_pylist()
        for kind, t in by_kind.items():
            if t["chunk_id"].to_pylist() != ids:
                raise ValueError(f"Shard {shard!r}: chunk_id của {kind!r} không khớp với chunks "
                                 f"(lô bắt đầu ở dòng {stats.rows}); dừng, không nạp.")
        toks = by_kind["tokens"]["tokenized_text"].to_pylist() if use_tokens else [None] * len(ids)
        rows = by_kind["chunks"].to_pylist()
        chunks = [Chunk(tokenized_text=tk, **r) for r, tk in zip(rows, toks)]
        vec_col = by_kind["vectors"]["vector"].combine_chunks()
        dim = vec_col.type.list_size
        vectors = vec_col.values.to_numpy(zero_copy_only=False).reshape(len(ids), dim)
        write(chunks, vectors, embedding_model=embedding_model, embedding_version=embedding_version)
        stats.rows += len(ids)
    stats.seconds = time.perf_counter() - t0
    stats.extra = {"mode": mode, "embedding_version": embedding_version}
    write_manifest(work, stats)
    return stats