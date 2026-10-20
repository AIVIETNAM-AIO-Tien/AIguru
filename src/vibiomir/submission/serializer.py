"""Competition output formatting. PROVISIONAL schema (from the overview doc, unverified):

    {"id": <query_id>, "relevant_docs": [<doc_id>...],
     "relevant_chunks": [{"doc_id": <doc_id>, "chunk_text": <text>}...]}
"""
from __future__ import annotations

from ..models import RetrievalResult


def to_submission_record(result: RetrievalResult) -> dict:
    """TODO: build the record from RetrievalResult. Do not add fields absent from the official schema."""
    raise NotImplementedError("TODO: serialization after the schema is verified")


def write_submission(records: list[dict], path: str) -> None:
    raise NotImplementedError("TODO: JSON writer (check official file format: single array vs JSONL)")
