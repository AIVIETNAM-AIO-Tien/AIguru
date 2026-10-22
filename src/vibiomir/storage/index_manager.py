from __future__ import annotations

import logging

from ..config import LanceDBConfig
from .lancedb_store import LanceDBClient, sql_quote

log = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "vector_idx"


class IndexManager:
    def __init__(self, client: LanceDBClient, cfg: LanceDBConfig):
        self.client, self.cfg = client, cfg

    @property
    def fts_index_name(self) -> str:
        return f"fts_{self.cfg.fts_column}"

    # ----- tạo index ----------------------------------------------------------------------

    def create_fts_index(self) -> None:
        """Tạo full-text (BM25) index theo cấu hình `fts_*` trong config.
        Luôn drop index cũ trước: với `replace=True`, LanceDB vẫn giữ tham số tokenizer cũ
        """
        self._drop_if_exists(self.fts_index_name)
        table = self.client.open_table()     
        table.create_fts_index(
            self.cfg.fts_column,
            base_tokenizer=self.cfg.fts_base_tokenizer,
            lower_case=self.cfg.fts_lower_case,
            stem=self.cfg.fts_stem,
            remove_stop_words=self.cfg.fts_remove_stop_words,
            ascii_folding=self.cfg.fts_ascii_folding,
            name=self.fts_index_name,
        )
        log.info("Đã tạo FTS index %s trên cột %s", self.fts_index_name, self.cfg.fts_column)

    def create_vector_index(self) -> None:
        """Tạo vector index ANN (nhanh hơn nhưng có thể mất recall so với exact search).
        Chưa dùng ở pilot. Khi dùng, so recall của ANN với exact search trên cùng tập truy vấn.
        """
        self._drop_if_exists(VECTOR_INDEX_NAME)
        table = self.client.open_table()    
        kwargs = {}
        if self.cfg.vector_num_partitions:
            kwargs["num_partitions"] = self.cfg.vector_num_partitions
        table.create_index(metric=self.cfg.metric, index_type=self.cfg.vector_index_type,
                           vector_column_name="vector", name=VECTOR_INDEX_NAME, **kwargs)
        log.info("Đã tạo vector index %s (%s, %s)", VECTOR_INDEX_NAME,
                 self.cfg.vector_index_type, self.cfg.metric)

    # ----- kiểm tra / xóa -----------------------------------------------------------------

    def list_index_names(self) -> list[str]:
        return [i.name for i in self.client.open_table().list_indices()]

    def needs_rebuild(self, version_tag: str, dimension: int) -> bool:
        """True nếu dữ liệu/index trong bảng không còn khớp với model đang dùng.

        - Số chiều của cột vector khác `dimension`, hoặc
        - có dòng có `embedding_version` khác `version_tag` (vector trộn nhiều phiên bản).
        Với bảng đã đổi số chiều thì phải tạo lại bảng (create_table(overwrite=True)), không chỉ index.
        """
        table = self.client.open_table()
        if table.schema.field("vector").type.list_size != dimension:
            return True
        return table.count_rows(f"embedding_version != {sql_quote(version_tag)}") > 0

    def drop_indexes(self) -> None:
        for name in self.list_index_names():
            self.client.open_table().drop_index(name)    

    def _drop_if_exists(self, name: str) -> None:
        if name in self.list_index_names():
            self.client.open_table().drop_index(name)