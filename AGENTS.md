# Repository Guidelines

## Project Shape

- `vipii` is a Python package using a `src/` layout: package code lives in `src/vipii`.
- The public API is exported from `src/vipii/__init__.py`.
- Core modules:
  - `models.py`: frozen dataclasses for `PIIMatch` and regex `Pattern`.
  - `recognizers/`: Vietnamese recognizers. `pattern.py` subclasses Presidio's
    `PatternRecognizer`; `ner.py` configures Presidio's `HuggingFaceNerRecognizer`;
    `validators.py` holds the `valid_*` checks.
  - `recognizers/builtin/`: one module per built-in entity, each exposing `recognizer()`.
  - `config.py`: YAML loading for user-supplied recognizer configs only.
  - `scoring.py`: context-window normalization and score boosting.
  - `detector.py`: `PIIDetector` over Presidio's `AnalyzerEngine`, plus overlap resolution.
  - `cli.py`: `argparse` CLI for `vipii scan`.
  - `presidio.py`: engine wiring — context enhancer, registry and analyzer builders.
  - `nlp.py`: `NlpEngine` implementations and the `create_nlp_engine` factory.

## Coding Style

- Target Python is `>=3.10`; keep compatibility with Python 3.10 through 3.13.
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

- Presidio is the core. vipii recognizers *are* `presidio_analyzer` recognizers; there is no
  adapter layer, and `PIIDetector` delegates to `AnalyzerEngine.analyze()`.
- Built-in recognizers are regex `Pattern` objects plus optional validators and context words.
- Each built-in lives in its own module under `recognizers/builtin/`, named after the
  recognizer, exposing a `recognizer()` factory that returns a fresh instance. Register a new
  one by adding it to `BUILTIN_MODULES` in `recognizers/builtin/__init__.py`; a test asserts
  every module in the package is registered.
- Validators map to `invalidate_result()`, never `validate_result()`: a truthy `validate_result`
  forces the score to `MAX_SCORE` and would discard base scores and context boosting.
- Pattern regexes are compiled with `regex.IGNORECASE` only. Presidio defaults to `I|M|S`, which
  would change what `.`, `^` and `$` match.
- Context boosting lives in `PatternRecognizer.enhance_using_context()`, not Presidio's global
  `LemmaContextAwareEnhancer`, because context words are per pattern and token windows are per
  recognizer. The global enhancer is disabled via `VipiiContextAwareEnhancer`.
- NLP engines are registered in `NLP_ENGINES` and built by `create_nlp_engine()`; `underthesea` is
  the default. Add new engines by subclassing `VipiiNlpEngine` and implementing `tokenize()`.
- Engine tokens feed `scoring.context_window()` via `spans=`, so the tokenizer determines how far a
  token window reaches. Tokenizers without offsets must realign against the source text.
- An empty registry must keep `NoopEntityRecognizer`: `AnalyzerEngine` loads Presidio's US/EU
  default recognizers whenever `registry.recognizers` is falsy.
- `PIIDetector` exposes three lazily built engines: `analyzer` (everything), `pattern_analyzer`
  and `ner_analyzer`. The NER strategies need the split because Presidio cannot restrict
  `analyze()` to part of a registry; `pattern_analyzer` reuses `analyzer` when no NER
  recognizer has to be excluded. `add_recognizer()` invalidates all three.
- `PIIDetector.detect()` resolves cross-entity overlaps that Presidio's `remove_duplicates` leaves.
- vipii does not implement NER. `recognizers/ner.py` only configures Presidio's
  `HuggingFaceNerRecognizer` with Vietnamese label defaults via `ner_recognizer()`.
- NER strategies classify recognizers via `is_ner_recognizer()`, which matches Presidio's
  model-backed recognizer classes. Set `vipii_is_ner` on a recognizer to override; tests and
  examples use that to stand in for a model without transformers and torch.
- `PIIDetector.redact()` delegates to `presidio-anonymizer`.
- Optional dependencies (`ner`, `spark`) should stay lazy.

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

CI runs the same Ruff and pytest commands on Python 3.10 and 3.13.
