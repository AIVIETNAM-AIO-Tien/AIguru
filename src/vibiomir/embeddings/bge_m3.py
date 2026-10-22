"""Embedding dense bằng BAAI/bge-m3 (qua thư viện FlagEmbedding).

Cách dùng:
    model = BgeM3EmbeddingModel(settings.embedding)
    vecs  = model.encode_documents(["đoạn văn 1", "đoạn văn 2"])   # (2, 1024)
    q     = model.encode_query("triệu chứng tiểu đường")           # (1024,)

Model chỉ được tải ở lần encode đầu tiên; tạo object không chạm mạng.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import EmbeddingModel

log = logging.getLogger(__name__)


class BgeM3EmbeddingModel(EmbeddingModel):
    """Chỉ lấy dense vector. Sparse / ColBERT của bge-m3 chưa dùng (cố ý tắt).

    bge-m3 không cần prefix cho query hay document, nên hai loại dùng chung một đường code.
    `backend` chỉ dành cho test (truyền model giả); khi chạy thật để mặc định None.
    """

    def __init__(self, cfg: EmbeddingConfig, backend: Any = None):
        self.cfg = cfg
        self._model = backend  # None cho tới lần encode đầu tiên

    # ----- thông tin mô tả model (không cần tải model) ------------------------------------

    model_name = property(lambda self: self.cfg.model_name)
    revision = property(lambda self: self.cfg.revision)
    normalized = property(lambda self: self.cfg.normalize)
    dimension = property(lambda self: self.cfg.dimension)

    @property
    def version_tag(self) -> str:
        """Chuỗi ghi vào cột `embedding_version`: gồm mọi thứ làm thay đổi vector.
        Hai lần index có version_tag khác nhau thì KHÔNG được trộn vector với nhau."""
        c = self.cfg
        return (f"{c.model_name}@{c.revision or 'unpinned'}"
                f"|fp16={c.use_fp16}|norm={c.normalize}|maxlen={c.max_length}")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Trả mảng float32 (len(texts), dimension), cùng thứ tự với `texts`.

        Nên truyền batch vừa phải (vài nghìn chunk); FlagEmbedding tự chia nhỏ theo
        `batch_size` và tự sắp theo độ dài bên trong.
        """
        if isinstance(texts, str):
            raise TypeError("encode_documents cần một danh sách chuỗi, không phải một chuỗi đơn")
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Trả mảng float32 (dimension,)."""
        if not isinstance(text, str):
            raise TypeError("encode_query cần một chuỗi")
        return self._encode([text])[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Gọi model, rồi kiểm tra và chuẩn hóa kết quả."""
        self._load_model()
        out = self._model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)

        if vecs.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Kích thước vector {vecs.shape}, kỳ vọng {(len(texts), self.dimension)}. "
                f"Nếu model có số chiều khác, sửa embedding.dimension trong config "
                f"(và dựng lại index).")
        if not np.isfinite(vecs).all():
            raise ValueError("Vector chứa NaN/inf")

        if self.cfg.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Có vector toàn 0, không thể chuẩn hóa")
            vecs = vecs / norms
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def _load_model(self) -> None:
        """Tải model một lần (lazy). Có revision thì tải đúng snapshot đó."""
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError("Cần FlagEmbedding: pip install 'vibiomir[embedding]'") from exc

        model_path = self._resolve_model_path()
        log.info("Tải model từ %s (fp16=%s)", model_path, self.cfg.use_fp16)
        self._model = BGEM3FlagModel(
            model_path, use_fp16=self.cfg.use_fp16, normalize_embeddings=self.cfg.normalize)

    def _resolve_model_path(self) -> str:
        if not self.cfg.revision:
            log.warning("embedding.revision chưa được pin: kết quả có thể không tái lập được")
            return self.cfg.model_name
        from huggingface_hub import snapshot_download
        return snapshot_download(self.cfg.model_name, revision=self.cfg.revision)