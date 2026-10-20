from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..config import CorpusConfig, ErrorPolicyConfig
from .profiling import DomainInventory

log = logging.getLogger(__name__)


class MatchStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING_FILE = "missing_file"
    READ_ERROR = "read_error"


@dataclass
class DomainValidationRow:
    domain: str
    language: str
    expected_url_count: int
    actual_row_count: Optional[int]
    difference: Optional[int]            # actual - expected
    status: MatchStatus
    file_path: Optional[str]
    error: Optional[str] = None


@dataclass
class ValidationReport:
    rows: list[DomainValidationRow] = field(default_factory=list)
    unexpected_files: list[str] = field(default_factory=list)   # Parquet with no profiling domain

    def by_status(self, status: MatchStatus) -> list[DomainValidationRow]:
        return [r for r in self.rows if r.status == status]


def expected_path(parquet_dir: str | Path, domain: str, pattern: str) -> Path:
    return Path(parquet_dir) / pattern.format(domain=domain)


def count_parquet_rows(path: str | Path) -> int:
    """Row count from Parquet footer metadata only (no data is loaded)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def compare_row_counts(domain: str, language: str, expected: int, actual: Optional[int],
                       file_path: Optional[str], error: Optional[str] = None) -> DomainValidationRow:
    """Pure comparison logic (easy to unit test)."""
    if error is not None:
        return DomainValidationRow(domain, language, expected, None, None,
                                   MatchStatus.READ_ERROR, file_path, error)
    if actual is None:
        return DomainValidationRow(domain, language, expected, None, None,
                                   MatchStatus.MISSING_FILE, file_path)
    diff = actual - expected
    status = MatchStatus.MATCH if diff == 0 else MatchStatus.MISMATCH
    return DomainValidationRow(domain, language, expected, actual, diff, status, file_path)


def validate_row_counts(inventory: DomainInventory, parquet_dir: str | Path,
                        corpus: CorpusConfig, policy: ErrorPolicyConfig,
                        languages: Optional[list[str]] = None) -> ValidationReport:
    """Validate every inventory domain (optionally filtered by language).

    Mismatches are recorded, not fatal, unless policy says "raise". Output order is
    sorted by domain for reproducibility.
    """
    parquet_dir = Path(parquet_dir)
    report = ValidationReport()
    wanted = sorted((r for r in inventory.records if not languages or r.language in languages),
                    key=lambda r: r.domain)
    expected_files: set[Path] = set()
    for rec in wanted:
        path = expected_path(parquet_dir, rec.domain, corpus.filename_pattern)
        expected_files.add(path)
        if not path.exists():
            row = compare_row_counts(rec.domain, rec.language, rec.expected_url_count, None, str(path))
        else:
            try:
                row = compare_row_counts(rec.domain, rec.language, rec.expected_url_count,
                                         count_parquet_rows(path), str(path))
            except Exception as exc:  # noqa: BLE001 - recorded in report
                row = compare_row_counts(rec.domain, rec.language, rec.expected_url_count,
                                         None, str(path), error=repr(exc))
        report.rows.append(row)
        _apply_policy(row, policy)

    if parquet_dir.is_dir():
        # Unexpected = any parquet not matching ANY inventory domain (all languages).
        known = {expected_path(parquet_dir, r.domain, corpus.filename_pattern)
                 for r in inventory.records}
        report.unexpected_files = sorted(str(p) for p in parquet_dir.glob("*.parquet")
                                         if p not in known)
    return report


def _apply_policy(row: DomainValidationRow, policy: ErrorPolicyConfig) -> None:
    rule = {MatchStatus.MISSING_FILE: policy.on_missing_file,
            MatchStatus.MISMATCH: policy.on_row_count_mismatch,
            MatchStatus.READ_ERROR: policy.on_read_error}.get(row.status)
    if rule is None:
        return
    msg = f"{row.domain}: {row.status.value} (expected={row.expected_url_count}, actual={row.actual_row_count})"
    if rule == "raise":
        raise RuntimeError(msg)
    log.warning(msg)


def find_duplicate_ids_and_urls(path: str | Path, id_col: str, url_col: str) -> dict[str, int]:
    """TODO: read ONLY id/url columns (pyarrow `columns=`) and count duplicates per file.

    Cross-file duplicate detection needs a global strategy (hashing / DuckDB later).
    """
    raise NotImplementedError("TODO: duplicate id/url detection")
