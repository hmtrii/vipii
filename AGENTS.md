# Repository Guidelines

## Project Shape

- `vipii` is a Python package using a `src/` layout: package code lives in `src/vipii`.
- The public API is exported from `src/vipii/__init__.py`.
- Core modules:
  - `models.py`: frozen dataclasses for `PIIMatch` and regex `Pattern`.
  - `recognizers.py`: built-in Vietnamese structured PII recognizers, validators, and registry.
  - `scoring.py`: context-window normalization and score boosting.
  - `detector.py`: detector orchestration, overlap resolution, and redaction.
  - `cli.py`: `argparse` CLI for `vipii scan`.
  - `presidio.py`: optional Presidio adapter; importing the module should not require Presidio.

## Coding Style

- Target Python is `>=3.9`; keep compatibility with Python 3.9 through 3.13.
- Use `from __future__ import annotations` in Python modules.
- Prefer small, typed functions and dataclasses over large classes.
- Use absolute imports from `vipii`, matching the existing modules.
- Keep source formatted for Ruff with a 100-character line length.
- Existing lint rules come from Ruff: `E`, `F`, `I`, `UP`, `B`, and `SIM`.
- Keep user-facing text and file IO UTF-8 friendly; tests and examples contain Vietnamese text.
- Use Any|Any for optional return in funciton instead of typing.Optional
- At interface functions, place '...' instead left empty

## Naming Patterns

- PII entity labels are uppercase strings such as `CCCD`, `PHONE_NUMBER`, and `BANK_ACCOUNT`.
- Recognizer names are lowercase snake_case such as `phone_number` and `vehicle_plate`.
- Validators use `valid_*` names and return `bool`.
- Helper functions use snake_case and are module-level unless they need object state.
- CLI command functions are named around the action, for example `scan_input`, `scan_file`,
  and `scan_text`.

## Architecture Patterns

- Built-in recognizers are regex `Pattern` objects plus optional validators and context words.
- Scores start from `base_score` and are boosted by nearby context words in `scoring.py`.
- `PIIDetector.detect()` gathers candidates from the registry, then resolves overlapping spans.
- `PIIDetector.redact()` masks detected spans while preserving surrounding text.
- Custom patterns are added through `PIIDetector.add_pattern()` and wrapped as a recognizer.
- Optional dependencies should stay lazy, as in `presidio.py`.

## Testing Style

- Tests use pytest and live under `tests/`.
- Prefer behavior-focused tests against the public API or CLI entry points.
- CLI tests call `vipii.cli.main(...)` directly and assert stdout with `capsys`.
- Use `tmp_path` for file-based CLI tests.
- Fixture-driven detector coverage uses JSONL in `tests/fixtures/`.
- Some tests use `# type: ignore[no-untyped-def]` for pytest fixtures without annotations.

## Commands

Install for development:

```bash
pip install -e ".[dev]"
```

Run lint and format checks:

```bash
ruff check .
ruff format --check .
```

Run tests:

```bash
pytest
```

CI runs the same Ruff and pytest commands on Python 3.9 and 3.13.
