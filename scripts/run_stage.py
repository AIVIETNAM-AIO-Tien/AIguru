"""Chạy một giai đoạn offline trên một hoặc nhiều domain.

Ví dụ (pilot nhỏ, lấy ngẫu nhiên 200 tài liệu mỗi domain, seed cố định):
    python scripts/run_stage.py --config configs/baseline.yaml --stage chunk    --max-docs 200 --seed 0
    python scripts/run_stage.py --config configs/baseline.yaml --stage embed
    python scripts/run_stage.py --config configs/baseline.yaml --stage tokenize
    python scripts/run_stage.py --config configs/baseline.yaml --stage index --build-fts

Chạy toàn bộ: bỏ --max-docs. Mặc định xử lý các domain thuộc corpus.languages (hiện là vi);
--domains a.vn,b.vn để chọn riêng. Shard đã xong sẽ bị bỏ qua (dùng --force để làm lại).
"""
import argparse
import logging

from vibiomir.config import load_settings, setup_logging
from vibiomir.data.parquet_reader import ParquetCorpusReader
from vibiomir.data.profiling import read_profiling
from vibiomir.pipeline.indexing_pipeline import (WorkDir, run_chunk, run_embed, run_index,
                                                 run_tokenize)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--stage", required=True, choices=["chunk", "embed", "tokenize", "index"])
ap.add_argument("--domains", default=None, help="danh sách domain, cách nhau bằng dấu phẩy")
ap.add_argument("--max-docs", type=int, default=None, help="chỉ lấy ngẫu nhiên N tài liệu mỗi domain")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force", action="store_true")
ap.add_argument("--index-mode", default="upsert", choices=["upsert", "add"])
ap.add_argument("--build-fts", action="store_true", help="(stage index) tạo FTS index sau khi nạp")
a = ap.parse_args()

s = load_settings(a.config)
setup_logging(s.logging)
log = logging.getLogger("run_stage")
inv = read_profiling(s.paths.profiling_csv, s.profiling)
wanted = a.domains.split(",") if a.domains else None
recs = [r for r in inv.records
        if (r.domain in wanted if wanted else r.language in s.corpus.languages)]
if not recs:
    raise SystemExit("Không có domain nào được chọn")
work, B = WorkDir(s.paths.work_dir), s.pipeline.batch_rows

if a.stage == "chunk":
    from vibiomir.chunking.fixed_size import FixedSizeTokenChunker
    reader = ParquetCorpusReader(s.paths.parquet_dir, s.corpus, s.errors)
    chunker = FixedSizeTokenChunker(s.chunking)
    fn = lambda r: run_chunk(r, reader, chunker, work, batch_rows=B, max_docs=a.max_docs,
                             seed=a.seed, force=a.force)
elif a.stage == "embed":
    from vibiomir.embeddings.bge_m3 import BgeM3EmbeddingModel
    embedder = BgeM3EmbeddingModel(s.embedding)
    fn = lambda r: run_embed(r.domain, embedder, work, batch_rows=B, force=a.force)
elif a.stage == "tokenize":
    from vibiomir.text import vi_tokenize
    fn = lambda r: run_tokenize(r.domain, vi_tokenize, work, batch_rows=B, force=a.force)
else:
    from vibiomir.embeddings.bge_m3 import BgeM3EmbeddingModel
    from vibiomir.storage.index_manager import IndexManager
    from vibiomir.storage.lancedb_store import ChunkRepository, LanceDBClient
    client = LanceDBClient(s.lancedb)
    model = BgeM3EmbeddingModel(s.embedding)
    if not client.table_exists():
        client.create_table(model.dimension)
    repo = ChunkRepository(client)
    if IndexManager(client, s.lancedb).needs_rebuild(model.version_tag, model.dimension) and repo.count():
        raise SystemExit("Bảng LanceDB chứa vector từ phiên bản/số chiều khác với model hiện tại; "
                         "tạo lại bảng (create_table(overwrite=True)) trước khi nạp.")
    fn = lambda r: run_index(r.domain, repo, work, embedding_model=model.model_name,
                             embedding_version=model.version_tag, batch_rows=B, mode=a.index_mode)

tot_rows = tot_sec = 0
for r in sorted(recs, key=lambda r: r.domain):
    st = fn(r)
    if st.skipped:
        log.info("%-28s bỏ qua (đã có)", r.domain)
        continue
    tot_rows += st.rows; tot_sec += st.seconds
    log.info("%-28s %8d dòng  %8.1fs  %9.1f dòng/s", r.domain, st.rows, st.seconds, st.rows_per_sec)
log.info("TỔNG: %d dòng, %.1fs", tot_rows, tot_sec)

if a.stage == "index" and a.build_fts:
    IndexManager(client, s.lancedb).create_fts_index()