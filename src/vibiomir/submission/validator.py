"""Submission validation. PROVISIONAL: keys mirror the unverified overview doc."""
from __future__ import annotations

PROVISIONAL_KEYS = {"id", "relevant_docs", "relevant_chunks"}


def validate_record(rec: dict) -> list[str]:
    """Return a list of problems (empty = passes the PROVISIONAL structural check only)."""
    problems: list[str] = []
    if set(rec) != PROVISIONAL_KEYS:
        problems.append(f"keys {sorted(rec)} != {sorted(PROVISIONAL_KEYS)}")
        return problems
    docs = rec["relevant_docs"]
    if len(docs) != len(set(docs)):
        problems.append("duplicate ids in relevant_docs")
    for i, ch in enumerate(rec["relevant_chunks"]):
        if set(ch) != {"doc_id", "chunk_text"}:
            problems.append(f"chunk {i}: unexpected keys {sorted(ch)}")
    # TODO: replace with the official validator once available.
    return problems
