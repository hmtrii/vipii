# Changelog

## 0.2.0

Presidio is now the detection core. This release is **not backward compatible**.

### Changed

- `presidio-analyzer` and `presidio-anonymizer` are required dependencies. The optional
  `vipii[presidio]` extra is gone; `pip install vipii[presidio]` will fail.
- vipii recognizers are now Presidio recognizers. `vipii.recognizers.PatternRecognizer`
  subclasses `presidio_analyzer.PatternRecognizer`. The `VipiiEntityRecognizer` adapter is
  removed.
- `vipii.recognizers.Recognizer` is now an alias of `presidio_analyzer.EntityRecognizer`.
  Custom recognizers must subclass it instead of implementing a bare `recognize()` method.
- `PIIDetector.redact()` is backed by `presidio-anonymizer`.
- Minimum Python is 3.10.

### Added

- Pluggable NLP engines with a `create_nlp_engine()` factory and an `NLP_ENGINES` registry.
  `underthesea` is the default and a required dependency; `PIIDetector(nlp_engine="regex")`
  selects the dependency-free tokenizer, and any `NlpEngine` instance is accepted.
- Context scoring now uses the active engine's tokens, so Vietnamese compounds count as one
  token inside a recognizer's token window.

- `PIIDetector.detect()` accepts Presidio's analysis options: `entities`, `score_threshold`,
  `allow_list`, `allow_list_match`, `ad_hoc_recognizers`, `context`, and
  `return_decision_process`.
- `PIIDetector.detect_batch()` backed by Presidio's `BatchAnalyzerEngine`, and
  `vipii.spark.detect_batch()` for DataFrame workloads.
- `PIIDetector.redact(operators=...)` exposes Presidio operators (`replace`, `hash`,
  `encrypt`, ...).
- `PIIDetector.analyzer`, `pattern_analyzer` and `ner_analyzer` properties, the
  `pattern_recognizers` / `ner_recognizers` views, `PIIDetector.analyze()` for raw Presidio
  results, and `PIIDetector.supported_entities()`.
- `vipii.ner_recognizer()` builds Presidio's `HuggingFaceNerRecognizer` with Vietnamese
  label defaults and forwards extra keyword arguments (`chunk_size`, `device`, ...).
- Presidio's model-backed recognizers (`HuggingFaceNerRecognizer`, `SpacyRecognizer`,
  `GLiNERRecognizer`) participate in the NER strategies. Set `vipii_is_ner` on a recognizer
  to override the classification.

### Removed

- `src/vipii/builtin_recognizers.yml`. The built-in recognizers are now one Python module
  each under `vipii.recognizers.builtin`, registered in `BUILTIN_MODULES`. Definitions are
  unchanged: names, labels, validators, token windows, regexes, context words, base scores
  and flags all match the YAML exactly. `config.load_builtin_recognizers()` is gone;
  `load_recognizers_from_yaml()` still loads user configs.

- `max_workers`. Presidio's `AnalyzerEngine` evaluates recognizers sequentially, so the
  detector no longer runs an internal thread pool. Parallelise across texts instead, or use
  `detect_batch()`.
- `vipii.recognizers.RecognizerRegistry`, which shadowed Presidio's class of the same name.
- `vipii.NERRecognizer` and its `pipeline_factory` hook. vipii no longer implements NER;
  `PIIDetector(ner_model=...)` and `add_ner_model()` now build Presidio's
  `HuggingFaceNerRecognizer`. Because Presidio requires `transformers` and `torch` in its
  constructor, `PIIDetector(ner_model=...)` now raises `ImportError` without the `[ner]`
  extra rather than deferring until the model is first used. The NER strategies are
  unchanged.
- `VipiiPresidioRecognizer`, which nested one `AnalyzerEngine` inside another.

### Fixed

- The analyzer engine is built once per recognizer group and cached. It was previously
  rebuilt on every `detect()` call, and once per segment or sentence under the `uncovered`
  and `chunked` NER strategies.
- The internal Nexus index moved from `pyproject.toml` to `uv.toml` so it is no longer
  published in the wheel's metadata.

### Compatibility

Detection output is unchanged from 0.1.3 with `nlp_engine="regex"`: labels, spans, scores,
recognizer names, and redacted text are byte-identical across the fixture dataset, validator
edge cases, and custom patterns.

With the default `underthesea` engine, one fixture changes: `MST` in
`"Mã số thuế chi nhánh 0312345678-001"` scores 0.70 rather than 0.50, because word segmentation
brings the `Mã số thuế` cue inside that recognizer's 3-token context window. No labels or spans
change.
