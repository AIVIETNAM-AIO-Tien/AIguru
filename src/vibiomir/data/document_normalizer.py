from __future__ import annotations

from typing import Any, Mapping, Optional

from ..config import CorpusConfig, ErrorPolicyConfig
from ..models import Document


class DocumentNormalizer:
    """Policies (documented, configurable):
    - `id` is preserved as str; a missing id raises (no synthetic ids).
    - missing/blank title -> None; missing text -> "" and flagged in provenance.
    - on_empty_text="keep" keeps the Document (flagged); "drop" returns None. The caller must
      count dropped rows so nothing disappears silently.
    - domain falls back to the profiling domain when the row lacks it.
    """

    def __init__(self, corpus: CorpusConfig, policy: ErrorPolicyConfig):
        self.cols = corpus.columns
        self.policy = policy

    @staticmethod
    def _clean(v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float) and v != v):  # None / NaN
            return None
        s = str(v).strip()
        return s or None

    def normalize(self, row: Mapping[str, Any], *, domain: str, language: str,
                  source_file: Optional[str] = None, row_index: Optional[int] = None) -> Optional[Document]:
        c = self.cols
        doc_id = self._clean(row.get(c["id"]))
        if doc_id is None:
            raise ValueError(f"row {row_index} in {source_file}: missing id (synthetic ids are not allowed)")
        text = self._clean(row.get(c["text"])) or ""
        empty = text == ""
        if empty and self.policy.on_empty_text == "drop":
            return None
        return Document(
            document_id=doc_id,
            source_url=self._clean(row.get(c["url"])) or "",
            domain=self._clean(row.get(c["domain"])) or domain,
            language=language,
            title=self._clean(row.get(c["title"])),
            text=text,
            provenance={"source_file": source_file, "row_index": row_index, "empty_text": empty},
        )
