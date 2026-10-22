"""Xử lý văn bản dùng chung giữa indexing và truy vấn.
"""
from __future__ import annotations

def vi_tokenize(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi: 'bệnh tiểu đường' -> 'bệnh tiểu_đường'. Import muộn."""
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(text)