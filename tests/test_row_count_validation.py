import pyarrow as pa, pyarrow.parquet as pq
import pytest
from vibiomir.config import CorpusConfig, ErrorPolicyConfig
from vibiomir.data.profiling import DomainInventory, DomainRecord, ProfilingReport
from vibiomir.data.validation import MatchStatus, compare_row_counts, validate_row_counts


def test_compare():
    assert compare_row_counts("d", "vi", 3, 3, "f").status == MatchStatus.MATCH
    r = compare_row_counts("d", "vi", 3, 5, "f")
    assert (r.status, r.difference) == (MatchStatus.MISMATCH, 2)
    assert compare_row_counts("d", "vi", 3, None, "f").status == MatchStatus.MISSING_FILE
    assert compare_row_counts("d", "vi", 3, None, "f", error="boom").status == MatchStatus.READ_ERROR


def _inv():
    recs = [DomainRecord("a.vn", "vi", "", 2), DomainRecord("b.vn", "vi", "", 1), DomainRecord("c.vn", "vi", "", 1)]
    return DomainInventory(recs, ProfilingReport())


def test_validate_dir(tmp_path):
    pq.write_table(pa.table({"id": ["1", "2"]}), tmp_path / "a.vn.parquet")
    pq.write_table(pa.table({"id": ["1", "2", "3"]}), tmp_path / "b.vn.parquet")
    pq.write_table(pa.table({"id": ["1"]}), tmp_path / "zzz.parquet")
    rep = validate_row_counts(_inv(), tmp_path, CorpusConfig(), ErrorPolicyConfig())
    st = {r.domain: r.status for r in rep.rows}
    assert st == {"a.vn": MatchStatus.MATCH, "b.vn": MatchStatus.MISMATCH, "c.vn": MatchStatus.MISSING_FILE}
    assert len(rep.unexpected_files) == 1


def test_raise_policy(tmp_path):
    with pytest.raises(RuntimeError):
        validate_row_counts(_inv(), tmp_path, CorpusConfig(), ErrorPolicyConfig(on_missing_file="raise"))
