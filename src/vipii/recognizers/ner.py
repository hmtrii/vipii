"""Optional model-backed NER recognizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vipii.models import PIIMatch

PipelineFactory = Callable[..., Callable[[str], list[dict[str, Any]]]]

DEFAULT_ENTITY_LABELS = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "NAME": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "ADDRESS",
    "ADDR": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
}


@dataclass
class NERRecognizer:
    """Wrap a Hugging Face token-classification pipeline as a vipii recognizer."""

    model_name: str
    name: str = "ner"
    min_score: float = 0.5
    aggregation_strategy: str = "simple"
    device: int = -1
    entity_label_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENTITY_LABELS))
    pipeline_factory: PipelineFactory | None = None
    _pipeline: Callable[[str], list[dict[str, Any]]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be provided for NERRecognizer")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")

    def recognize(self, text: str) -> list[PIIMatch]:
        matches = []
        for entity in flatten_entities(self.pipeline()(text)):
            match = self.match_from_entity(text, entity)
            if match:
                matches.append(match)
        return matches

    def pipeline(self) -> Callable[[str], list[dict[str, Any]]]:
        if self._pipeline is None:
            factory = self.pipeline_factory or load_transformers_pipeline
            self._pipeline = factory(
                "token-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy=self.aggregation_strategy,
                device=self.device,
            )
        return self._pipeline

    def match_from_entity(self, text: str, entity: dict[str, Any]) -> PIIMatch | None:
        raw_label = normalize_entity_label(str(entity.get("entity_group") or entity.get("entity")))
        label = self.entity_label_map.get(raw_label, raw_label)
        score = float(entity.get("score", 0))
        start = entity.get("start")
        end = entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 0 or end <= start or end > len(text) or score < self.min_score:
            return None
        return PIIMatch(
            label=label,
            start=start,
            end=end,
            text=text[start:end],
            score=score,
            recognizer=self.name,
        )


def load_transformers_pipeline(*args: Any, **kwargs: Any) -> Callable[[str], list[dict[str, Any]]]:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError("Install vipii[ner] to use NERRecognizer") from exc
    return pipeline(*args, **kwargs)


def normalize_entity_label(label: str) -> str:
    label = label.upper()
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label


def flatten_entities(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, list) for item in value):
        flattened = []
        for item in value:
            flattened.extend(flatten_entities(item))
        return flattened
    return [item for item in value if isinstance(item, dict)]
