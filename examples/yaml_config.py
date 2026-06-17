"""Load custom recognizers from a YAML file.

Run from the repository after installing the package:

    python examples/yaml_config.py
"""

from __future__ import annotations

from pathlib import Path

from _helpers import print_matches

from vipii import PIIDetector

HERE = Path(__file__).resolve().parent


def main() -> None:
    detector = PIIDetector.from_yaml(HERE / "custom_recognizers.yml")
    text = "Customer id KH-654321 cần xác minh tài khoản ngân hàng 123456789012."

    print_matches("Custom pattern from YAML", text, detector)


if __name__ == "__main__":
    main()
