"""Helpers for exercising NER without transformers or torch."""

from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

ENTITIES = {
    "Nguyễn Văn A": "PERSON",
    "Hà Nội": "LOCATION",
}


class StubNerRecognizer(EntityRecognizer):
    """A model-backed recognizer stand-in.

    ``vipii_is_ner`` opts it into the detector's NER strategies without pulling in
    Presidio's Hugging Face recognizer, which requires transformers and torch at
    construction time.
    """

    vipii_is_ner = True

    def __init__(self, name: str = "ner", seen_texts: list[str] | None = None) -> None:
        self.seen_texts = seen_texts if seen_texts is not None else []
        super().__init__(
            supported_entities=sorted(set(ENTITIES.values())),
            name=name,
            supported_language="vi",
        )

    def load(self) -> None:
        """No resources are required."""

    def analyze(self, text, entities, nlp_artifacts=None):  # type: ignore[no-untyped-def]
        del entities, nlp_artifacts
        self.seen_texts.append(text)
        results = []
        for phrase, label in ENTITIES.items():
            start = text.find(phrase)
            if start >= 0:
                results.append(
                    RecognizerResult(
                        entity_type=label,
                        start=start,
                        end=start + len(phrase),
                        score=0.98 if label == "PERSON" else 0.91,
                    )
                )
        return results


class ExplodingNerRecognizer(StubNerRecognizer):
    """Fails if the detector ever runs it."""

    def analyze(self, text, entities, nlp_artifacts=None):  # type: ignore[no-untyped-def]
        raise AssertionError("NER should not have run")


def presidio_hf_recognizer(entities: list[dict[str, object]], **kwargs):  # type: ignore[no-untyped-def]
    """Build Presidio's Hugging Face recognizer with a stub pipeline.

    transformers and torch are optional extras, so the module-level sentinels are
    replaced to allow construction without them.
    """
    import presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer as module

    class _Torch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    module.torch = _Torch
    module.hf_pipeline = lambda *args, **kwargs: None

    from vipii import ner_recognizer

    recognizer = ner_recognizer("fake-vietnamese-ner", **kwargs)
    recognizer.ner_pipeline = lambda text: entities
    return recognizer
