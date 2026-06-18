# vipii

`vipii` is a Python library for detecting Vietnamese personally identifiable information (PII) using regex-based and NER-based recognizers.

## Install

```bash
pip install vipii
```

For local development:

```bash
pip install -e ".[dev]"
```

## Python API

```python
from vipii import PIIDetector, Pattern

detector = PIIDetector()
detector.add_pattern(
    Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b", context_words=["mã khách hàng"])
)

matches = detector.detect(
    "Khách hàng Nguyễn Văn A, số điện thoại 0912 345 678, CCCD 001203000123."
)

for match in matches:
    print(match.label, match.text, match.score)
```

## Concurrent scanning

`PIIDetector.detect()` runs recognizers concurrently by default when the detector has more than one
recognizer. Use `max_workers` to cap the internal recognizer thread pool, or set `max_workers=1` to
force sequential recognition:

```python
from vipii import PIIDetector

detector = PIIDetector(max_workers=4)
matches = detector.detect("Số điện thoại 0912 345 678 và CCCD 001203000123")
```

When scanning many independent texts, you can run calls to `detect()` concurrently from your own
executor. Configure the detector before starting workers, then treat it as read-only while scans are
running; do not call `add_pattern()`, `add_recognizer()`, or `add_ner_model()` concurrently with
detection.

```python
from concurrent.futures import ThreadPoolExecutor

from vipii import PIIDetector

texts = [
    "Khách hàng A có số điện thoại 0912 345 678.",
    "Khách hàng B có CCCD 001203000123.",
]
detector = PIIDetector(max_workers=1)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(detector.detect, texts))
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

Build and inspect the package before uploading:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then upload the same checked artifacts to PyPI:

```bash
python -m twine upload dist/*
```
