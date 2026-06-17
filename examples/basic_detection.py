"""Detect built-in Vietnamese structured PII.

Run from the repository after installing the package:

    python examples/basic_detection.py
"""

from __future__ import annotations

from _helpers import print_matches

from vipii import PIIDetector


def main() -> None:
    detector = PIIDetector()
    text = (
        "Khách hàng có số điện thoại 0912 345 678, CCCD 001203000123, "
        "mã số thuế 0312345678 và biển số xe 51F-123.45."
    )

    print_matches("Built-in structured PII", text, detector)


if __name__ == "__main__":
    main()
