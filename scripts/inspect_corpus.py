"""Summarize profiling CSV (domain inventory, URL counts per language)."""
import argparse, json, sys
from vibiomir.config import load_settings
from vibiomir.data.profiling import read_profiling

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--csv", default=None)
a = ap.parse_args()
s = load_settings(a.config)
inv = read_profiling(a.csv or s.paths.profiling_csv, s.profiling)
print(json.dumps({"domains": len(inv.records),
                  "url_counts_by_language": inv.url_counts_by_language(),
                  "total_urls": sum(r.expected_url_count for r in inv.records),
                  "problems": inv.report.problems}, ensure_ascii=False, indent=2))
