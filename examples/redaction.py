"""Redact detected PII spans.

Run from the repository after installing the package:

    python examples/redaction.py
"""

from __future__ import annotations

from vipii import PIIDetector


def main() -> None:
    detector = PIIDetector()
    text = "Vui lòng gọi 0912345678 hoặc kiểm tra CCCD 001203000123."

    print("\nRedaction")
    print("---------")
    print(detector.redact(text))
    print(detector.redact(text, mask="x"))


if __name__ == "__main__":
    main()
