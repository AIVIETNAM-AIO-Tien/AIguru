import numpy as np
import pytest

pytest.importorskip("lancedb")

from vibiomir.config import LanceDBConfig
from vibiomir.models import Chunk
from vibiomir.storage.index_manager import IndexManager
from vibiomir.storage.lancedb_store import ChunkRepository, LanceDBClient, sql_quote

DIM = 4


def mk(i, text, tok=None, doc=None):
    return Chunk(chunk_id=f"d{i}::0", document_id=doc or f"d{i}", chunk_index=0, chunk_text=text,
                 title=None, source_url=f"u{i}", domain="x.vn", language="vi", token_count=3,
                 tokenized_text=tok or text)


@pytest.fixture
def env(tmp_path):
    cfg = LanceDBConfig(uri=str(tmp_path / "db"))
    client = LanceDBClient(cfg)
    client.create_table(DIM)
    return cfg, client, ChunkRepository(client), IndexManager(client, cfg)


def kw(version="v1"):
    return dict(embedding_model="m", embedding_version=version)


def load3(repo):
    chunks = [mk(1, "bệnh tiểu đường", "bệnh tiểu_đường"), mk(2, "cao huyết áp", "cao huyết_áp"),
              mk(3, "điều trị tiểu đường", "điều_trị tiểu_đường")]
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    repo.add(chunks, vecs, **kw())


def test_create_table_guard(env):
    _, client, _, _ = env
    with pytest.raises(FileExistsError):
        client.create_table(DIM)
    client.create_table(DIM, overwrite=True)           # chủ động ghi đè thì được


def test_open_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        LanceDBClient(LanceDBConfig(uri=str(tmp_path / "x"))).open_table()


def test_add_count_and_get_by_ids(env):
    _, _, repo, _ = env
    load3(repo)
    assert repo.count() == 3
    got = repo.get_by_ids(["d1::0", "d2::0", "nope"])
    assert {g["chunk_id"] for g in got} == {"d1::0", "d2::0"}
    assert "vector" not in got[0] and "chunk_text" in got[0]
    assert "vector" in repo.get_by_ids(["d1::0"], include_vector=True)[0]
    assert repo.get_by_ids([]) == []


def test_input_validation(env):
    _, _, repo, _ = env
    with pytest.raises(ValueError):                    # sai số chiều
        repo.add([mk(1, "a")], np.ones((1, 3), dtype=np.float32), **kw())
    with pytest.raises(ValueError):                    # NaN
        repo.add([mk(1, "a")], np.full((1, DIM), np.nan, dtype=np.float32), **kw())
    with pytest.raises(ValueError):                    # chunk_id trùng trong lô
        repo.add([mk(1, "a"), mk(1, "b")], np.ones((2, DIM), dtype=np.float32), **kw())
    assert repo.add([], np.zeros((0, DIM)), **kw()) == 0 and repo.count() == 0


def test_upsert_is_idempotent_and_updates(env):
    _, _, repo, _ = env
    load3(repo)
    new = [mk(1, "SỬA d1"), mk(9, "mới d9")]
    vecs = np.eye(DIM, dtype=np.float32)[:2]
    assert repo.upsert(new, vecs, **kw("v2")) == 2
    assert repo.upsert(new, vecs, **kw("v2")) == 2     # chạy lại không nhân đôi
    assert repo.count() == 4
    row = repo.get_by_ids(["d1::0"])[0]
    assert row["chunk_text"] == "SỬA d1" and row["embedding_version"] == "v2"


def test_delete_document_and_quote(env):
    _, _, repo, _ = env
    repo.add([mk(1, "a", doc="o'brien"), mk(2, "b")], np.eye(DIM, dtype=np.float32)[:2], **kw())
    repo.delete_document("o'brien")                    # id có dấu nháy đơn
    assert repo.count() == 1
    assert sql_quote("a'b") == "'a''b'"


def test_fts_keeps_vietnamese_diacritics(env):
    _, client, repo, idx = env
    load3(repo)
    idx.create_fts_index()
    table = client.open_table()
    search = lambda q: [r["chunk_id"] for r in table.search(q, query_type="fts").limit(5).to_list()]
    assert set(search("tiểu_đường")) == {"d1::0", "d3::0"}
    assert search("tieu_duong") == []                  # không bỏ dấu
    assert search("tiểu") == []                        # whitespace: không tách ở dấu gạch dưới
    idx.create_fts_index()                             # tạo lại không lỗi
    assert idx.list_index_names().count("fts_tokenized_text") == 1


def test_vector_search_cosine_exact(env):
    _, client, repo, _ = env
    load3(repo)
    r = client.open_table().search([1, 0, 0, 0]).metric("cosine").limit(3).to_list()
    assert [x["chunk_id"] for x in r] == ["d1::0", "d3::0", "d2::0"]
    assert r[0]["_distance"] == pytest.approx(0.0, abs=1e-6)       # distance = 1 - cosine


def test_needs_rebuild(env):
    _, _, repo, idx = env
    load3(repo)
    assert idx.needs_rebuild("v1", DIM) is False
    assert idx.needs_rebuild("v2", DIM) is True        # khác version
    assert idx.needs_rebuild("v1", 1024) is True       # khác số chiều


def test_vector_index_and_drop(tmp_path):
    cfg = LanceDBConfig(uri=str(tmp_path / "db"), vector_index_type="IVF_FLAT", vector_num_partitions=2)
    client = LanceDBClient(cfg); client.create_table(8)
    repo, idx = ChunkRepository(client), IndexManager(client, cfg)
    n = 300
    chunks = [mk(i, "t") for i in range(n)]
    repo.add(chunks, np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32), **kw())
    idx.create_vector_index(); idx.create_vector_index()           # tạo lại không lỗi
    assert idx.list_index_names() == ["vector_idx"]
    idx.drop_indexes()
    assert idx.list_index_names() == []