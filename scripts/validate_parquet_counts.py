"""Compare profiling url_count with actual Parquet row counts (footer metadata only)."""
import argparse, csv, sys
from dataclasses import asdict
from vibiomir.config import load_settings, setup_logging
from vibiomir.data.profiling import read_profiling
from vibiomir.data.validation import validate_row_counts

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--out", default=None, help="optional CSV report path")
ap.add_argument("--all-languages", action="store_true")
a = ap.parse_args()
s = load_settings(a.config)
setup_logging(s.logging)
inv = read_profiling(s.paths.profiling_csv, s.profiling)
rep = validate_row_counts(inv, s.paths.parquet_dir, s.corpus, s.errors,
                          None if a.all_languages else s.corpus.languages)
rows = [{**asdict(r), "status": r.status.value} for r in rep.rows]
if a.out and rows:
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["domain"], r["status"], r["expected_url_count"], r["actual_row_count"])
print("unexpected files:", rep.unexpected_files)
