"""BgeM3EmbeddingModel with fake backends only: no model download, no network."""
import sys
import types

import numpy as np
import pytest

from vibiomir.config import EmbeddingConfig
from vibiomir.embeddings.bge_m3 import BgeM3EmbeddingModel


class FakeBackend:
    def __init__(self, dim=8, scale=3.0):
        self.dim, self.scale, self.calls = dim, scale, []

    def encode(self, texts, **kw):
        self.calls.append((list(texts), kw))
        rng = np.random.default_rng(0)
        return {"dense_vecs": rng.normal(size=(len(texts), self.dim)) * self.scale + 0.1}


def make(**cfg):
    cfg.setdefault("dimension", 8)
    fb = FakeBackend()
    return BgeM3EmbeddingModel(EmbeddingConfig(**cfg), backend=fb), fb


def test_instantiation_is_lazy():
    m = BgeM3EmbeddingModel(EmbeddingConfig())
    assert m._model is None and m.dimension == 1024
    assert "FlagEmbedding" not in sys.modules


def test_documents_shape_dtype_and_norm():
    m, fb = make()
    v = m.encode_documents(["a", "b", "c"])
    assert v.shape == (3, 8) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_no_normalization_when_disabled():
    m, _ = make(normalize=False)
    v = m.encode_documents(["a", "b"])
    assert not np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert m.normalized is False


def test_query_shape_and_dense_only_kwargs():
    m, fb = make(batch_size=5, max_length=514)
    q = m.encode_query("xin chào")
    assert q.shape == (8,) and q.dtype == np.float32
    _, kw = fb.calls[-1]
    assert kw == {"batch_size": 5, "max_length": 514, "return_dense": True,
                  "return_sparse": False, "return_colbert_vecs": False}


def test_empty_input_and_type_errors():
    m, _ = make()
    assert m.encode_documents([]).shape == (0, 8)
    with pytest.raises(TypeError):
        m.encode_documents("a single string")
    with pytest.raises(TypeError):
        m.encode_query(["not", "a", "str"])


def test_bad_backend_outputs_raise():
    class Bad:
        def __init__(self, arr): self.arr = arr
        def encode(self, texts, **kw): return {"dense_vecs": self.arr}
    # sai số chiều, NaN, sai số chiều (1-D), vector toàn 0
    for arr in (np.ones((1, 3)), np.full((1, 4), np.nan), np.ones(4), np.zeros((1, 4))):
        m = BgeM3EmbeddingModel(EmbeddingConfig(dimension=4), backend=Bad(arr))
        with pytest.raises(ValueError):
            m.encode_documents(["only one"])


def test_dimension_mismatch_message_points_to_config():
    m = BgeM3EmbeddingModel(EmbeddingConfig(dimension=16), backend=FakeBackend(dim=8))
    with pytest.raises(ValueError, match="embedding.dimension"):
        m.encode_query("x")


def test_version_tag_changes_with_config():
    a = BgeM3EmbeddingModel(EmbeddingConfig()).version_tag
    b = BgeM3EmbeddingModel(EmbeddingConfig(revision="abc")).version_tag
    c = BgeM3EmbeddingModel(EmbeddingConfig(use_fp16=False)).version_tag
    assert len({a, b, c}) == 3 and "unpinned" in a and "@abc" in b


def test_load_passes_expected_kwargs(monkeypatch):
    seen = {}

    class FakeFlag:
        def __init__(self, path, **kw):
            seen.update(path=path, **kw)
        def encode(self, texts, **kw):
            return {"dense_vecs": np.ones((len(texts), 4), dtype=np.float32)}

    mod = types.ModuleType("FlagEmbedding"); mod.BGEM3FlagModel = FakeFlag
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    m = BgeM3EmbeddingModel(EmbeddingConfig(use_fp16=False, dimension=4))
    assert m.encode_query("x").shape == (4,)
    assert seen == {"path": "BAAI/bge-m3", "use_fp16": False, "normalize_embeddings": True}


def test_missing_flagembedding_gives_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    with pytest.raises(ImportError, match=r"vibiomir\[embedding\]"):
        BgeM3EmbeddingModel(EmbeddingConfig()).encode_query("x")