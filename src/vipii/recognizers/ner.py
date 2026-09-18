"""Model-backed NER recognizers.

vipii does not implement NER itself. This module configures Presidio's
``HuggingFaceNerRecognizer`` with Vietnamese defaults; ``transformers`` and
``torch`` come from the ``vipii[ner]`` extra and are required at construction
time by Presidio.
"""

from __future__ import annotations

from typing import Any

from presidio_analyzer import EntityRecognizer

from vipii.nlp import LANGUAGE

# Presidio's DEFAULT_LABEL_MAPPING lacks the address labels Vietnamese NER
# models commonly emit.
VIETNAMESE_LABEL_MAPPING = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "NAME": "PERSON",
    "PS": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "LC": "LOCATION",
    "ADDRESS": "ADDRESS",
    "ADDR": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "OG": "ORGANIZATION",
}


def ner_recognizer(
    model_name: str,
    *,
    min_score: float = 0.5,
    name: str = "ner",
    label_mapping: dict[str, str] | None = None,
    **kwargs: Any,
) -> EntityRecognizer:
    """Build a Presidio Hugging Face NER recognizer for Vietnamese.

    :param model_name: Hugging Face token-classification model.
    :param min_score: Minimum confidence, passed to Presidio as ``threshold``.
    :param label_mapping: Model label to entity label map.
    Remaining keyword arguments are forwarded to ``HuggingFaceNerRecognizer``.
    """
    if not model_name:
        raise ValueError("model_name must be provided for a NER recognizer")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")

    from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

    return HuggingFaceNerRecognizer(
        model_name=model_name,
        name=name,
        supported_language=LANGUAGE,
        label_mapping=dict(label_mapping or VIETNAMESE_LABEL_MAPPING),
        threshold=min_score,
        **kwargs,
    )
