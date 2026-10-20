"""Read the domain-level profiling CSV (NOT the document corpus)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import ProfilingConfig

# Verified labels look like "Tiếng Việt (VI)"; the ISO-ish code is in parentheses.
_LANG_RE = re.compile(r"\(([A-Za-z]{2,3})\)\s*$")


@dataclass(frozen=True)
class DomainRecord:
    domain: str
    language: str            # normalized: "vi" | "en" | "zh" (or "unknown")
    language_label: str      # original label, kept for provenance
    expected_url_count: int


@dataclass
class ProfilingReport:
    problems: list[str] = field(default_factory=list)
    duplicate_domains: list[str] = field(default_factory=list)
    invalid_count_domains: list[str] = field(default_factory=list)
    missing_domain_rows: int = 0


@dataclass
class DomainInventory:
    records: list[DomainRecord]
    report: ProfilingReport

    def by_domain(self) -> dict[str, DomainRecord]:
        return {r.domain: r for r in self.records}

    def url_counts_by_language(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.language] = out.get(r.language, 0) + r.expected_url_count
        return out


def parse_language(label: str) -> str:
    """'Tiếng Việt (VI)' -> 'vi'. Unrecognized labels -> 'unknown' (reported, not guessed)."""
    m = _LANG_RE.search(str(label).strip())
    return m.group(1).lower() if m else "unknown"


def read_profiling(path: str | Path, cfg: ProfilingConfig) -> DomainInventory:
    """Read, validate and project profiling CSV into a domain inventory.

    Raises ValueError if a mapped column is absent. Duplicates, invalid counts and
    missing domains are recorded in the report and rows are NOT silently dropped
    (duplicates: first occurrence is kept in `records`, all are listed in the report).
    """
    df = pd.read_csv(path, encoding=cfg.encoding, dtype=str)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    cols = cfg.columns
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise ValueError(f"profiling CSV missing columns {missing}; found {list(df.columns)}")

    report = ProfilingReport()
    records: list[DomainRecord] = []
    seen: set[str] = set()
    for i, row in df.iterrows():
        domain = (row[cols["domain"]] or "").strip() if isinstance(row[cols["domain"]], str) else ""
        if not domain:
            report.missing_domain_rows += 1
            report.problems.append(f"row {i}: empty domain")
            continue
        label = str(row[cols["language"]])
        lang = parse_language(label)
        if lang == "unknown":
            report.problems.append(f"{domain}: unrecognized language label {label!r}")
        try:
            count = int(str(row[cols["url_count"]]).replace(",", "").strip())
            if count < 0:
                raise ValueError
        except ValueError:
            report.invalid_count_domains.append(domain)
            report.problems.append(f"{domain}: invalid url_count {row[cols['url_count']]!r}")
            continue
        if domain in seen:
            report.duplicate_domains.append(domain)
            report.problems.append(f"{domain}: duplicate domain row")
            continue
        seen.add(domain)
        records.append(DomainRecord(domain, lang, label, count))
    return DomainInventory(records, report)
