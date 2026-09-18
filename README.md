# vipii

`vipii` is a Python 3.10+ library for detecting Vietnamese personally identifiable information
(PII). Microsoft Presidio provides the analyzer and recognizer registry, while vipii adds Vietnamese
regex recognizers, validators, context scoring, and optional NER.

## Install

```bash
pip install vipii
```

For local development:

```bash
pip install -e ".[dev]"
```

For Spark DataFrame support:

```bash
pip install "vipii[spark]"
```

## Python API

```python
from vipii import PIIDetector, Pattern

detector = PIIDetector()
detector.add_pattern(
    Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b", context_words=["mã khách hàng"])
)

text = "Khách hàng Nguyễn Văn A, số điện thoại 0912 345 678, CCCD 001203000123."
matches = detector.detect(text)

for match in matches:
    print(match.label, match.text, match.score)
```

## Presidio core

Presidio *is* the detection core. vipii's recognizers subclass `presidio_analyzer` classes
directly, so there is no adapter layer: `PatternRecognizer` extends Presidio's
`PatternRecognizer`, NER comes from Presidio's own recognizers, and `PIIDetector` delegates to
`AnalyzerEngine.analyze()` for Vietnamese (`vi`).

```python
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from vipii import PIIDetector

detector = PIIDetector()
assert isinstance(detector.analyzer, AnalyzerEngine)
assert all(isinstance(r, PatternRecognizer) for r in detector.registry.recognizers)
```

What vipii keeps on top of Presidio: Vietnamese regex patterns, checksum and range validators
(mapped to `invalidate_result()` so base scores survive), per-pattern context boosting with
per-recognizer token windows, and cross-entity overlap resolution.

### Built-in recognizers

Each built-in entity is one module under `vipii/recognizers/builtin/`, exposing a
`recognizer()` factory:

```python
from vipii.recognizers.builtin import built_in_recognizers, cccd

cccd.recognizer()  # just this one
built_in_recognizers()  # all fifteen
```

Add a new one by dropping a module in that package and listing it in `BUILTIN_MODULES`.
YAML is still supported for user-supplied recognizers via `--config` and
`PIIDetector.from_yaml()`.

### NLP engine

Presidio needs an `NlpEngine` even though every recognizer is regex-based: it supplies the tokens
that context scoring measures its window against. vipii ships two, selected by name:

| Engine | Default | Notes |
| --- | --- | --- |
| `underthesea` | yes | Segments Vietnamese compounds — `điện thoại` and `khách hàng` are one token each |
| `regex` | no | Dependency-free Unicode word split; one token per syllable |

```python
PIIDetector()  # underthesea
PIIDetector(nlp_engine="regex")  # no underthesea needed
PIIDetector(nlp_engine=MyEngine())  # any NlpEngine instance
```

Word segmentation matters because token windows are small. `MST` uses a 3-token window, so in
`"Mã số thuế chi nhánh 0312345678-001"` the cue `Mã số thuế` only falls inside the window when
`Mã số` and `chi nhánh` are segmented as compounds — scoring that match 0.70 with `underthesea`
against 0.50 with `regex`.

No spaCy model is involved; Presidio publishes no pretrained Vietnamese pipeline.

### Analysis options

`detect()` forwards Presidio's analysis options:

```python
detector = PIIDetector()

detector.detect(text, entities=["PHONE_NUMBER", "CCCD"])  # restrict entity types
detector.detect(text, score_threshold=0.8)  # drop low-confidence spans
detector.detect(text, allow_list=["0900000000"])  # never flag these values
detector.detect(text, return_decision_process=True)  # keep analysis explanations
```

### NER recognizers

vipii does not implement NER. `ner_model=` builds Presidio's `HuggingFaceNerRecognizer`
with Vietnamese label defaults:

```python
from vipii import PIIDetector

detector = PIIDetector(ner_model="NlpHUST/ner-vietnamese-electra-base", ner_min_score=0.7)
```

This needs the `vipii[ner]` extra (`transformers` and `torch`), which Presidio requires at
construction time — `PIIDetector(ner_model=...)` raises `ImportError` without them.

Any Presidio `EntityRecognizer` can be passed directly instead, and `vipii.ner_recognizer()`
forwards extra keyword arguments (`chunk_size`, `device`, `aggregation_strategy`, ...) to
Presidio:

```python
from vipii import PIIDetector, ner_recognizer

detector = PIIDetector(
    recognizers=[ner_recognizer("NlpHUST/ner-vietnamese-electra-base", device=0)]
)
```

vipii keeps the NER *strategies* (`always`, `fallback`, `uncovered`, `chunked`, `never`), which
control when a model runs relative to the pattern recognizers. Presidio's model-backed
recognizers are detected automatically; set `vipii_is_ner` on a recognizer to override the
classification.

### Batch scanning

`detect_batch()` runs Presidio's `BatchAnalyzerEngine` over many texts:

```python
detector = PIIDetector()
results = detector.detect_batch(
    [
        "Khách hàng A có số điện thoại 0912 345 678.",
        "Khách hàng B có CCCD 001203000123.",
    ]
)
```

`PIIDetector` holds no per-call state, so you can also submit `detect()` to your own executor.
Configure the detector first, then treat it as read-only while scans are running — do not call
`add_pattern()`, `add_recognizer()`, or `add_ner_model()` while scans are in flight.

### Redaction

`redact()` is backed by `presidio-anonymizer` and masks every character of each span by default.
Pass `operators` to use any Presidio operator:

```python
from presidio_anonymizer.entities import OperatorConfig

detector.redact(text)  # 'Số điện thoại **********.'
detector.redact(text, operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})})
```


## PySpark

The Spark adapter is optional and keeps PySpark imports lazy. It can add detected matches or
redacted text to a DataFrame text column:

```python
from vipii.spark import with_pii_matches, with_redacted_column

df = spark.createDataFrame(
    [("Số CCCD của tôi là 001203000123",)],
    ["text"],
)

matches_df = with_pii_matches(df, input_col="text", output_col="pii_matches")
redacted_df = with_redacted_column(df, input_col="text", output_col="redacted")
```

## Optional NER

Regex recognizers cover structured PII. For free-form names, locations, organizations, and addresses,
enable an external Hugging Face token-classification model:

```bash
pip install "vipii[ner]"
vipii scan "Nguyễn Văn A sống tại Hà Nội" --ner-model your-vietnamese-ner-model
```

```python
from vipii import PIIDetector

detector = PIIDetector(ner_model="your-vietnamese-ner-model")
matches = detector.detect("Nguyễn Văn A sống tại Hà Nội")
```

The NER layer maps model labels such as `PER`, `LOC`, and `ORG` to `PERSON`, `LOCATION`, and
`ORGANIZATION`. The model is not bundled; choose and evaluate one for your domain before production
use.

To reduce model inference cost, choose an NER strategy:

- `always`: run pattern recognizers and NER on the full text.
- `fallback`: run NER only when pattern recognizers find no structured PII.
- `uncovered`: run pattern recognizers first, then run NER only on text outside detected spans.
- `chunked`: split text into chunks, redact structured PII spans, then run NER on useful chunks.
- `never`: skip NER even if a model is configured.

```bash
vipii scan "Số điện thoại 0912345678" --ner-model your-vietnamese-ner-model --ner-strategy fallback
vipii scan "Số điện thoại 0912345678 của Nguyễn Văn A" --ner-model your-vietnamese-ner-model --ner-strategy uncovered
vipii scan "Số điện thoại 0912345678 của Nguyễn Văn A" --ner-model your-vietnamese-ner-model --ner-strategy chunked
```

```python
detector = PIIDetector(ner_model="your-vietnamese-ner-model", ner_strategy="fallback")
detector = PIIDetector(ner_model="your-vietnamese-ner-model", ner_strategy="uncovered")
detector = PIIDetector(ner_model="your-vietnamese-ner-model", ner_strategy="chunked")
```

## CLI

```bash
vipii scan "Số điện thoại 0912 345 678 và CCCD 001203000123"
vipii scan examples/customer_service.txt
vipii scan examples/customer_service.txt --format json
vipii scan examples/customer_service.txt --redact
vipii scan "CCCD 001203000123" --redact
vipii scan "Mã khách hàng KH-123456" --config examples/custom_recognizers.yml
vipii scan "Nguyễn Văn A sống tại Hà Nội" --ner-model your-vietnamese-ner-model
vipii scan "Số điện thoại 0912345678" --ner-model your-vietnamese-ner-model --ner-strategy fallback
vipii scan "Số điện thoại 0912345678 của Nguyễn Văn A" --ner-model your-vietnamese-ner-model --ner-strategy uncovered
vipii scan "Số điện thoại 0912345678 của Nguyễn Văn A" --ner-model your-vietnamese-ner-model --ner-strategy chunked
```

## YAML recognizer config

Built-in recognizers are loaded from `src/vipii/builtin_recognizers.yml`. You can append your own
recognizers from a YAML file without writing Python:

```yaml
recognizers:
  - name: customer_id
    label: CUSTOMER_ID
    patterns:
      - regex: '\bKH-\d{6}\b'
        context_words: ["mã khách hàng", "customer id"]
        base_score: 0.6
```

Use `validator` only when you want one of vipii's built-in validators: `cccd`, `cmnd`, `phone`,
`email_address`, `date_of_birth`, `tax_code`, `bank_card`, `bank_account`, `social_insurance`,
`health_insurance`, `passport`, `vehicle_plate`, `driver_license`, `ip_address`, or `device_id`.

## Built-in recognizers

- `CCCD` and `CMND`
- `PHONE_NUMBER`
- `EMAIL_ADDRESS`
- `DATE_OF_BIRTH`
- `MST`
- `SOCIAL_INSURANCE_NUMBER`
- `HEALTH_INSURANCE_NUMBER`
- `BANK_CARD`
- `BANK_ACCOUNT`
- `PASSPORT`
- `VEHICLE_PLATE`
- `DRIVER_LICENSE`
- `IP_ADDRESS`
- `DEVICE_ID`

The recognizers intentionally favor clear structured PII plus nearby Vietnamese context words such as
`số điện thoại`, `cccd`, `mã số thuế`, and `biển số xe`. Names and free-form addresses can be handled
by the optional NER layer.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## Publishing

Publishing is handled manually from GitHub Actions. On the `release` branch, run the `Publish`
workflow with **Run workflow** and enter the version to publish, for example `0.1.3`.

To inspect a package locally before publishing:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```
