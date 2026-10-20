# ViBioMIR baseline retrieval — skeleton

Status: **skeleton only**. Implemented: config, models, profiling reader, row-count validation,
document normalizer, Parquet streaming reader, Arrow chunk schema. Everything else raises
`NotImplementedError` (see `TODO`s).

Flow: `profing.csv` -> domain inventory -> per-domain Parquet -> `Document` -> `Chunk` (512/100 tokens)
-> bge-m3 vectors -> LanceDB (vector + FTS) -> dense + BM25 -> RRF -> chunks + documents -> submission.

Run: `pip install -e .[dev]` then `pytest`; `python scripts/inspect_corpus.py --csv path/to/profing.csv`.
