# Contributing

STAC-Mem welcomes focused bug reports, reproducible examples, and small pull requests.

## Development setup

```bash
python3 quickstart.py --dev
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests scripts quickstart.py
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Pull requests

- Add a regression test for behavior changes.
- Preserve source provenance and temporal semantics in new data models.
- Do not commit API keys, runtime databases, downloaded datasets, or generated runs.
- Keep network-backed integration tests separate from deterministic unit tests.

By contributing, you agree that your contribution is licensed under Apache-2.0.
