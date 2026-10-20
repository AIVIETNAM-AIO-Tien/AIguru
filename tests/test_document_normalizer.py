import pytest
import pyarrow as pa, pyarrow.parquet as pq
from vibiomir.config import CorpusConfig, ErrorPolicyConfig
from vibiomir.data.document_normalizer import DocumentNormalizer
from vibiomir.data.parquet_reader import ParquetCorpusReader
from vibiomir.data.profiling import DomainRecord


def n(policy="keep"):
    return DocumentNormalizer(CorpusConfig(), ErrorPolicyConfig(on_empty_text=policy))


def test_basic_and_empty_policy():
    row = {"id": 7, "url": "http://x", "title": "  ", "text": "hello"}
    d = n().normalize(row, domain="x.vn", language="vi")
    assert (d.document_id, d.title, d.domain) == ("7", None, "x.vn")
    empty = {"id": "8", "url": "u", "text": None}
    kept = n("keep").normalize(empty, domain="x.vn", language="vi")
    assert kept.text == "" and kept.provenance["empty_text"] is True
    assert n("drop").normalize(empty, domain="x.vn", language="vi") is None


def test_missing_id_raises():
    with pytest.raises(ValueError):
        n().normalize({"url": "u", "text": "t"}, domain="d", language="vi")


def test_reader_roundtrip(tmp_path):
    pq.write_table(pa.table({"id": ["1", "2"], "url": ["u1", "u2"], "domain": ["d.vn"] * 2,
                             "title": ["t", None], "text": ["a", ""]}), tmp_path / "d.vn.parquet")
    r = ParquetCorpusReader(tmp_path, CorpusConfig(), ErrorPolicyConfig())
    docs = list(r.iter_documents(DomainRecord("d.vn", "vi", "", 2)))
    assert [d.document_id for d in docs] == ["1", "2"] and docs[1].provenance["empty_text"]
