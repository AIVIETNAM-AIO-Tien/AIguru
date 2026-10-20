# ViBioMIR — TODO 

## 1. Đã làm

| Module | Nội dung | Trạng thái |
|---|---|---|
| `config.py`, `configs/baseline.yaml` | Dataclass + YAML, key lạ báo lỗi | Xong |
| `models.py` | `Document`, `Chunk`, `Query`, `Candidate`, `FusedCandidate`, `RankedDocument`, `RetrievalResult` | Xong |
| `data/profiling.py` | Đọc `profing.csv` (utf-8-sig), tách mã ngôn ngữ, báo trùng/sai số | Xong, đã kiểm trên CSV thật |
| `data/validation.py` | So số dòng Parquet (footer) với `url_count`, policy warn/raise, file thiếu/thừa | Xong, test trên Parquet giả |
| `data/document_normalizer.py` | Chuẩn hóa dòng -> `Document`, policy text rỗng/title thiếu, không sinh id | Xong |
| `data/parquet_reader.py` | Stream theo batch, kiểm cột bắt buộc | Xong, chưa thử trên Parquet thật |
| `storage/lancedb_store.py` | `chunk_schema()` (Arrow) | Xong; client/repository là TODO |
| `submission/validator.py` | Kiểm cấu trúc tạm thời | Xong (provisional) |
| `scripts/` | `inspect_corpus.py`, `validate_parquet_counts.py` | Xong |

Chỉ là interface, mọi hàm `raise NotImplementedError`: chunknig, BGE-M3, LanceDB client/repository/index, dense, BM25, RRF, hybrid, document ranker, hai pipeline, serializer.

## 2. Đã xác minh 

`profing.csv`: cột `ten_mien`, `ngon_ngu`, `so_luong_url`; 94 domain (vi 50, zh 43, en 1); tổng 1.220.598 URL (vi 740.598, en 240.000, zh 240.000).

## 3. Thứ tự implement

### Phase 0 — Xác minh dữ liệu 
1. Lấy schema + 3–5 dòng mẫu của 1–2 Parquet; liệt kê tên file.
2. Chỉnh `corpus.columns`, `filename_pattern`, `paths.*` trong config.
3. Chạy `scripts/validate_parquet_counts.py --out report.csv`; ghi lại mismatch.
4. Đọc thể lệ + (nếu có) code chấm điểm: trả lời các mục "chưa xác minh" ở mục 2.

### Phase 1 — Chunker
- `FixedSizeTokenChunker.chunk`: ưu tiên windowing trực tiếp bằng tokenizer bge-m3 với `offset_mapping` (đúng 512/stride 412, có `char_start/char_end`, không cắt bớt text). Phương án thay thế: `RecursiveCharacterTextSplitter.from_huggingface_tokenizer`, nhưng phải test độ dài thực.
- `chunk_id = document_id::chunk_index` ( `make_chunk_id`).
- Test: không mất text (ghép lại phủ toàn bộ), `token_count <= 512`, id ổn định, văn bản tiếng Việt có dấu.

### Phase 3 — Embedding + LanceDB pilot
- `BgeM3EmbeddingModel`: tải lazy, pin `revision`, fp16, batch theo độ dài, đọc `dimension` từ model, vector float32 (L2-normalize nếu `normalize`).
- `LanceDBClient`, `ChunkRepository` (add/upsert theo `chunk_id`), `IndexManager`. **Pin phiên bản lancedb** và đọc docs FTS/vector index của đúng bản đó trước khi viết.
- Pilot ~10k tài liệu: exact vector search (chưa ANN). Ghi `embedding_model`/`embedding_version` vào bảng; `needs_rebuild()` so sánh với model đang dùng.

### Phase 4 — Retrieval
1. `DenseRetriever`: encode query -> search; `score = 1 - cosine_distance` (cao hơn = tốt hơn); rank bắt đầu từ 1; giữ `_distance` trong `raw`.
2. Tokenizer tiếng Việt: dùng pyvi, lưu vào cột `tokenized_text`; FTS index trên cột này; query phải qua cùng tokenizer. Đánh giá có cải thiện so với `chunk_text` thô không.
3. `LanceDBBM25Retriever`.
4. `RRFFusion`: dedupe theo `chunk_id`, `1/(k+rank)` với rank từ 1, k=60, tie-break xác định; test bằng ví dụ tính tay.
5. `HybridRetriever`, `DocumentRanker` (best_chunk; không trùng `document_id`; giữ danh sách chunk nguyên vẹn).

### Phase 5 — Query pipeline + submission 
- `QueryPipeline.run`: normalize nhẹ -> hybrid -> cắt `final_chunk_k` -> gom doc -> cắt `final_doc_k`.
- `to_submission_record`, `write_submission`, validator theo schema chính thức; `chunk_text` luôn là văn bản gốc (không pyvi, không prefix `Title:`).
- Chạy end-to-end trên pilot, in điểm Phase 1 làm baseline đầu tiên.

### Phase 6 — Index toàn corpus tiếng Việt (chạy theo shard)
- Nếu dùng Modal: giữ lõi `vibiomir` không import `modal`; thêm `scripts/modal_app.py` mỏng; chia shard theo domain; mỗi shard ghi Parquet vector có tên cố định (idempotent/resume); một job duy nhất dựng bảng LanceDB và index; trọng số model nằm sẵn trong image/Volume; đặt giới hạn chi tiêu trước khi chạy.
- Quyết định index toàn bộ hay một phần sau khi có số liệu chi phí từ Phase 3. Cân nhắc ANN (IVF_PQ/HNSW) chỉ khi exact search quá chậm.

### Phase 7 — Cải thiện (theo thứ tự lợi ích kỳ vọng, mỗi mục phải đo)
1. Reranker cross-encoder (bge-reranker-v2-m3) trên top 50-100 chunk. 
2. Tối ưu K: K riêng cho doc và chunk, K động theo điểm (F2 ưu tiên recall).
3. Kích thước chunk và overlap; tìm kiếm trên `title`; embedding có/không title.
4. Ghép chunk cùng doc (chiến lược A/B/C): **chỉ làm sau khi xác nhận cách tính recall chunk và thể lệ cho phép**. Ghép theo `chunk_index`, loại phần overlap bằng `char_start/char_end`, đặt trần độ dài.
5. Cross-lingual: dịch query sang en/zh cho nhánh BM25 chỉ khi gold thực sự nằm ở en/zh; cần index riêng và tokenizer tiếng Trung.

