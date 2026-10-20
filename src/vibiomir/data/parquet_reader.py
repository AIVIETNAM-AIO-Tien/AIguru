from __future__ import annotations

from pathlib import Path
from typing import Iterator
import pyarrow.parquet as pq

from ..config import CorpusConfig, ErrorPolicyConfig
from ..models import Document
from .document_normalizer import DocumentNormalizer
from .profiling import DomainRecord
from .validation import expected_path


class ParquetCorpusReader:
    """Streams normalized Documents per domain in batches (never loads a whole corpus).
    """

    def __init__(self, parquet_dir: str | Path, corpus: CorpusConfig,
                 policy: ErrorPolicyConfig, batch_size: int = 2048):
        self.dir = Path(parquet_dir)
        self.corpus = corpus
        self.batch_size = batch_size
        self.normalizer = DocumentNormalizer(corpus, policy)

    def path_for(self, rec: DomainRecord) -> Path:
        return expected_path(self.dir, rec.domain, self.corpus.filename_pattern)

    def schema_of(self, rec: DomainRecord):
        return pq.ParquetFile(self.path_for(rec)).schema_arrow

    def iter_documents(self, rec: DomainRecord) -> Iterator[Document]:
        path = self.path_for(rec)
        wanted = list(dict.fromkeys(self.corpus.columns.values()))
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        absent = [c for c in wanted if c not in have]
        required_missing = [c for c in absent if c in (self.corpus.columns["id"],
                                                       self.corpus.columns["url"],
                                                       self.corpus.columns["text"])]
        if required_missing:
            raise ValueError(f"{path}: missing required columns {required_missing}")
        idx = 0
        for batch in pf.iter_batches(batch_size=self.batch_size, columns=[c for c in wanted if c in have]):
            for row in batch.to_pylist():
                doc = self.normalizer.normalize(row, domain=rec.domain, language=rec.language,
                                                source_file=str(path), row_index=idx)
                idx += 1
                if doc is not None:
                    yield doc
