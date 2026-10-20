import pytest
from vibiomir.config import ProfilingConfig
from vibiomir.data.profiling import parse_language, read_profiling

CSV = "\ufefften_mien,ngon_ngu,so_luong_url\na.vn,Tiếng Việt (VI),10\nb.com,Tiếng Anh (EN),5\na.vn,Tiếng Việt (VI),3\nc.cn,Tiếng Trung (ZH),x\n"


def test_parse_language():
    assert parse_language("Tiếng Việt (VI)") == "vi"
    assert parse_language("???") == "unknown"


def test_read_profiling(tmp_path):
    p = tmp_path / "p.csv"; p.write_text(CSV, encoding="utf-8")
    inv = read_profiling(p, ProfilingConfig())
    assert [r.domain for r in inv.records] == ["a.vn", "b.com"]
    assert inv.report.duplicate_domains == ["a.vn"]
    assert inv.report.invalid_count_domains == ["c.cn"]
    assert inv.url_counts_by_language() == {"vi": 10, "en": 5}


def test_missing_column_raises(tmp_path):
    p = tmp_path / "p.csv"; p.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_profiling(p, ProfilingConfig())
