## 1. Offline

### 1.1. 
```bash
git clone https://github.com/AIVIETNAM-AIO-Tien/AIguru.git
cd AIguru
```

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 1.2. 
```bash
pip install -r requirements.txt  
pip install -e . --no-deps          
```

- Cách nhanh, **không** tái lập (cài bản mới nhất): `pip install -e ".[dev,storage,embedding,vi,chunking]"`.

### 1.3. Kiểm tra môi trường
```bash
pytest          # chạy từ thư mục gốc repo; kỳ vọng tất cả pass
```
Chạy `pytest` từ thư mục khác thư mục gốc sẽ làm một test đọc `configs/baseline.yaml` bị lỗi đường dẫn.


## 2. Kiểm tra dữ liệu trước khi chạy

```bash
python scripts/inspect_corpus.py --config configs/baseline.yaml
python scripts/validate_parquet_counts.py --config configs/baseline.yaml --out report.csv
```

## 3. Run Offline pipeline

```bash
# 1. Chia chunk (CPU, cần tokenizer bge-m3 nên máy phải có mạng ở lần đầu)
python scripts/run_stage.py --config configs/baseline.yaml --stage chunk --max-docs 200 --seed 0

# 2a. Embedding (GPU)
python scripts/run_stage.py --config configs/baseline.yaml --stage embed

# 2b. Tách từ pyvi (CPU) - độc lập với 2a, có thể chạy song song
python scripts/run_stage.py --config configs/baseline.yaml --stage tokenize

# 3. Nạp vào LanceDB và tạo FTS index
python scripts/run_stage.py --config configs/baseline.yaml --stage index --build-fts
```

Chạy theo domain: thêm `--domains a.vn,b.vn`. Mặc định script xử lý mọi domain thuộc `corpus.languages` (hiện là `vi`).


## 4. Chạy toàn bộ

Bỏ `--max-docs`, giữ nguyên cấu hình:
```bash
python scripts/run_stage.py --config configs/baseline.yaml --stage chunk
python scripts/run_stage.py --config configs/baseline.yaml --stage embed
python scripts/run_stage.py --config configs/baseline.yaml --stage tokenize
python scripts/run_stage.py --config configs/baseline.yaml --stage index --build-fts
```
- Shard đã có file sẽ **bị bỏ qua**, nên có thể ngắt và chạy lại để tiếp tục.
- Muốn làm lại một giai đoạn: thêm `--force` (và nhớ làm lại các giai đoạn sau nó).
- Nếu đổi `chunk_size`, `chunk_overlap` hoặc tokenizer, chunk thay đổi nên phải làm lại **cả bốn** giai đoạn (xóa `work/` hoặc dùng `--force` ở mọi giai đoạn).
- Nếu đổi model embedding hoặc revision, làm lại `embed` và `index`, và tạo lại bảng LanceDB.

## 5. Cấu trúc thư mục làm việc

```
work/
  chunks/<domain>.parquet     chunk_id, document_id, chunk_index, chunk_text, title,
                              source_url, domain, language, token_count, char_start, char_end
  vectors/<domain>.parquet    chunk_id, vector (float32)
  tokens/<domain>.parquet     chunk_id, tokenized_text (pyvi)
  manifests/<stage>__<domain>.json
```
