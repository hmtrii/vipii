"""Use an optional Hugging Face NER model.

Run from the repository after installing the optional dependencies:

    VIPII_RUN_NER=1 python examples/ner_model.py

Override the default model when needed:

    VIPII_RUN_NER=1 VIPII_NER_MODEL=your-model python examples/ner_model.py
"""

from __future__ import annotations

import os

from _helpers import print_matches

from vipii import PIIDetector

DEFAULT_NER_MODEL = "NlpHUST/ner-vietnamese-electra-base"


def main() -> None:
    model_name = os.environ.get("VIPII_NER_MODEL", DEFAULT_NER_MODEL)
    if os.environ.get("VIPII_RUN_NER") != "1":
        print("\nOptional NER model")
        print("------------------")
        print(f"Default model: {model_name!r}")
        print("Set VIPII_RUN_NER=1 to load the model and run this example.")
        return

    detector = PIIDetector(ner_model=model_name, ner_min_score=0.7)
    text = "Nguyễn Văn A sống tại 123A tân thuận Hà Nội và làm việc ở Công ty ABC."

    try:
        print_matches("Optional NER model", text, detector)
    except ImportError as exc:
        print("\nOptional NER model")
        print("------------------")
        print(f"Skipped {model_name!r}: {exc}")


if __name__ == "__main__":
    main()
