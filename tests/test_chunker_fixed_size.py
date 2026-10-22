import re

import pytest

from vibiomir.chunking.fixed_size import FixedSizeTokenChunker
from vibiomir.config import ChunkingConfig
from vibiomir.models import Document


def fake_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False,
                   verbose=False):
    assert add_special_tokens is False and return_offsets_mapping is True and truncation is False
    return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


def doc(n_words, text=None):
    text = text if text is not None else " ".join(f"w{i}" for i in range(n_words))
    return Document("D1", "http://u", "x.vn", "vi", "Tiêu đề", text)


def chunker(size=512, overlap=100):
    return FixedSizeTokenChunker(ChunkingConfig(chunk_size=size, chunk_overlap=overlap), fake_tokenizer)


def test_default_windows_1000_tokens():
    cs = chunker().chunk(doc(1000))
    assert [c.token_count for c in cs] == [512, 512, 176]        # starts 0, 412, 824
    assert [c.chunk_id for c in cs] == ["D1::0", "D1::1", "D1::2"]


@pytest.mark.parametrize("n,expected", [(1, 1), (512, 1), (513, 2), (924, 2), (925, 3)])
def test_chunk_counts_and_no_tail_inside_overlap(n, expected):
    assert len(chunker().chunk(doc(n))) == expected


def test_text_is_original_and_offsets_consistent():
    d = doc(1000)
    for c in chunker().chunk(d):
        assert d.text[c.char_start:c.char_end] == c.chunk_text
        assert c.document_id == "D1" and c.source_url == "http://u" and c.title == "Tiêu đề"


def test_every_token_is_covered_and_overlap_is_100():
    d = doc(1000)
    cs = chunker().chunk(d)
    covered = set()
    for c in cs:
        covered |= set(c.chunk_text.split())
    assert covered == set(d.text.split())
    first, second = cs[0].chunk_text.split(), cs[1].chunk_text.split()
    assert first[-100:] == second[:100]


def test_blank_document_and_stability():
    assert chunker().chunk(doc(0, text="   \n ")) == []
    assert chunker().chunk(doc(700)) == chunker().chunk(doc(700))


def test_invalid_config():
    with pytest.raises(ValueError):
        chunker(size=100, overlap=100)


def test_module_imports_without_loading_transformers():
    import sys
    assert "vibiomir.chunking.fixed_size" in sys.modules   # import ở đầu file test đã thành công