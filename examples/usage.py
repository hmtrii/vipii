"""Run all vipii usage examples.

Run from the repository after installing the package:

    python examples/usage.py
"""

from __future__ import annotations

import basic_detection
import concurrent_detection
import custom_only
import custom_pattern
import ner_model
import redaction
import yaml_config


def main() -> None:
    basic_detection.main()
    concurrent_detection.main()
    redaction.main()
    custom_pattern.main()
    yaml_config.main()
    custom_only.main()
    ner_model.main()


if __name__ == "__main__":
    main()
