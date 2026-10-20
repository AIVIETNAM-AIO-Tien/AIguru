"""Imports, config, and explicit NotImplementedError for unimplemented behavior."""
import pytest
from vibiomir.config import ChunkingConfig, Settings, load_settings
from vibiomir.chunking.base import make_chunk_id
from vibiomir.chunking.fixed_size import FixedSizeTokenChunker
from vibiomir.embeddings.bge_m3 import BgeM3EmbeddingModel
from vibiomir.models import Document, Query
from vibiomir.retrieval.document_ranker import DocumentRanker
from vibiomir.retrieval.fusion import RRFFusion
from vibiomir.storage.lancedb_store import chunk_schema, LanceDBClient
from vibiomir.submission.validator import validate_record
import vibiomir.pipeline.indexing_pipeline, vibiomir.pipeline.query_pipeline  # noqa: F401


def test_config_yaml_matches_defaults():
    s = load_settings("configs/baseline.yaml")
    assert s.chunking.stride == 412 and s.retrieval.rrf_k == 60
    assert s.profiling.columns == Settings().profiling.columns


def test_unknown_config_key(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("chunking: {chunk_sze: 1}\n")
    with pytest.raises(KeyError):
        load_settings(p)


def test_chunk_id_stable():
    assert make_chunk_id("abc", 3) == make_chunk_id("abc", 3) == "abc::3"


def test_chunker_validates_and_is_unimplemented():
    with pytest.raises(ValueError):
        FixedSizeTokenChunker(ChunkingConfig(chunk_size=100, chunk_overlap=100))
    doc = Document("1", "u", "d", "vi", None, "t")
    with pytest.raises(NotImplementedError):
        FixedSizeTokenChunker(ChunkingConfig()).chunk(doc)


def test_unimplemented_fail_loudly():
    with pytest.raises(NotImplementedError):
        RRFFusion().fuse({})
    with pytest.raises(NotImplementedError):
        DocumentRanker().rank([])
    with pytest.raises(NotImplementedError):
        BgeM3EmbeddingModel(Settings().embedding).encode_query("x")  # no model download
    with pytest.raises(NotImplementedError):
        LanceDBClient(Settings().lancedb).connect()


def test_schema_and_submission_check():
    sch = chunk_schema(1024)
    assert "vector" in sch.names and "tokenized_text" in sch.names
    assert "tokenized_text" not in chunk_schema(8, with_tokenized_text=False).names
    ok = {"id": "q", "relevant_docs": ["a"], "relevant_chunks": [{"doc_id": "a", "chunk_text": "t"}]}
    assert validate_record(ok) == []
    assert validate_record({**ok, "relevant_docs": ["a", "a"]})
