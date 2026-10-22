import json
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lancedb")

from vibiomir.chunking.fixed_size import FixedSizeTokenChunker
from vibiomir.config import ChunkingConfig, CorpusConfig, ErrorPolicyConfig, LanceDBConfig
from vibiomir.data.parquet_reader import ParquetCorpusReader
from vibiomir.data.profiling import DomainRecord
from vibiomir.pipeline.indexing_pipeline import (WorkDir, iter_exact_batches, run_chunk, run_embed,
                                                 run_index, run_tokenize, sample_documents)
from vibiomir.storage.index_manager import IndexManager
from vibiomir.storage.lancedb_store import ChunkRepository, LanceDBClient

DIM = 4
REC = DomainRecord("a.vn", "vi", "", 4)


def fake_tok(text, **kw):
    return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


class FakeEmbedder:
    dimension, version_tag, model_name = DIM, "fake@v1", "fake"

    def encode_documents(self, texts):
        return np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)


def words(n, p="tiểu_đường"):
    return " ".join(f"{p}{i}" for i in range(n))


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"; corpus.mkdir()
    pq.write_table(pa.table({
        "id": ["1", "2", "3", "4"], "url": [f"u{i}" for i in range(4)], "domain": ["a.vn"] * 4,
        "title": ["t"] * 4, "text": [words(12), words(7), "   ", words(30)]}), corpus / "a.vn.parquet")
    reader = ParquetCorpusReader(corpus, CorpusConfig(), ErrorPolicyConfig())
    chunker = FixedSizeTokenChunker(ChunkingConfig(chunk_size=5, chunk_overlap=2), fake_tok)
    return reader, chunker, WorkDir(tmp_path / "work"), tmp_path


def test_chunk_stage_counts_manifest_and_resume(env):
    reader, chunker, work, _ = env
    st = run_chunk(REC, reader, chunker, work, batch_rows=3)
    t = pq.read_table(work.path("chunks", "a.vn"))
    assert t.num_rows == st.rows > 0
    assert st.extra["documents"] == 4 and st.extra["documents_without_chunks"] == 1   # doc rỗng được đếm
    assert json.loads(work.manifest("chunk", "a.vn").read_text())["rows"] == st.rows
    assert run_chunk(REC, reader, chunker, work).skipped is True                      # resume
    assert not list(work.root.rglob("*.tmp"))                                         # không còn file tạm


def test_failed_stage_leaves_no_partial_file(env):
    reader, chunker, work, _ = env
    class Boom(FixedSizeTokenChunker):
        def chunk(self, doc):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        run_chunk(REC, reader, Boom(ChunkingConfig(chunk_size=5, chunk_overlap=2), fake_tok), work)
    assert not work.path("chunks", "a.vn").exists() and not list(work.root.rglob("*.tmp"))


def test_embed_and_tokenize_are_row_aligned(env):
    reader, chunker, work, _ = env
    run_chunk(REC, reader, chunker, work)
    run_embed("a.vn", FakeEmbedder(), work, batch_rows=4)
    run_tokenize("a.vn", lambda s: s.upper(), work, batch_rows=3)
    ids = pq.read_table(work.path("chunks", "a.vn"))["chunk_id"].to_pylist()
    assert pq.read_table(work.path("vectors", "a.vn"))["chunk_id"].to_pylist() == ids
    assert pq.read_table(work.path("tokens", "a.vn"))["chunk_id"].to_pylist() == ids
    assert json.loads(work.manifest("embed", "a.vn").read_text())["version_tag"] == "fake@v1"


def full_run(env):
    reader, chunker, work, tmp = env
    run_chunk(REC, reader, chunker, work)
    run_embed("a.vn", FakeEmbedder(), work)
    run_tokenize("a.vn", lambda s: s.lower(), work)
    cfg = LanceDBConfig(uri=str(tmp / "db"))
    client = LanceDBClient(cfg); client.create_table(DIM)
    return work, client, ChunkRepository(client), IndexManager(client, cfg)


def test_index_stage_end_to_end(env):
    work, client, repo, idx = full_run(env)
    n = pq.read_table(work.path("chunks", "a.vn")).num_rows
    st = run_index("a.vn", repo, work, embedding_model="fake", embedding_version="fake@v1", batch_rows=4)
    assert repo.count() == n == st.rows
    run_index("a.vn", repo, work, embedding_model="fake", embedding_version="fake@v1")   # upsert: không nhân đôi
    assert repo.count() == n
    assert idx.needs_rebuild("fake@v1", DIM) is False
    row = repo.get_by_ids(["1::0"])[0]
    assert row["tokenized_text"] and row["chunk_text"].startswith("tiểu_đường0")        # văn bản gốc giữ nguyên
    idx.create_fts_index()
    hits = client.open_table().search("tiểu_đường0", query_type="fts").limit(10).to_list()
    assert hits and all("chunk_text" in h for h in hits)


def test_index_refuses_misaligned_shard(env):
    work, client, repo, _ = full_run(env)
    p = work.path("vectors", "a.vn")
    t = pq.read_table(p)
    pq.write_table(t.take(list(reversed(range(t.num_rows)))), p)                        # đảo thứ tự dòng
    with pytest.raises(ValueError, match="không khớp"):
        run_index("a.vn", repo, work, embedding_model="fake", embedding_version="fake@v1")
    assert repo.count() == 0                                                              # không nạp gì


def test_index_requires_all_files(env):
    reader, chunker, work, tmp = env
    run_chunk(REC, reader, chunker, work)
    cfg = LanceDBConfig(uri=str(tmp / "db")); client = LanceDBClient(cfg); client.create_table(DIM)
    with pytest.raises(FileNotFoundError):
        run_index("a.vn", ChunkRepository(client), work, embedding_model="m", embedding_version="v")


def test_sampling_is_reproducible_and_ordered(env):
    reader, _, _, _ = env
    docs = list(reader.iter_documents(REC))
    a = sample_documents(docs, 2, seed=1); b = sample_documents(docs, 2, seed=1)
    assert [d.document_id for d in a] == [d.document_id for d in b] and len(a) == 2
    assert [d.document_id for d in a] == sorted(d.document_id for d in a)               # giữ thứ tự gốc
    assert len(sample_documents(docs, 10, seed=0)) == 4


def test_exact_batches_ignore_row_groups(tmp_path):
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"v": list(range(10))}), p, row_group_size=3)
    sizes = [t.num_rows for t in iter_exact_batches(pq.ParquetFile(p), ["v"], 4)]
    assert sizes == [4, 4, 2]