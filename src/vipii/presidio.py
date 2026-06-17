"""Optional Presidio adapter."""

from __future__ import annotations

from vipii import PIIDetector


class VipiiPresidioRecognizer:
    """Presidio-compatible recognizer adapter.

    Importing this module does not require Presidio. Instantiating the adapter does.
    """

    def __init__(self, supported_entities: list[str] | None = None) -> None:
        try:
            from presidio_analyzer import EntityRecognizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional extra
            raise ImportError("Install vipii[presidio] to use VipiiPresidioRecognizer") from exc

        class _Recognizer(EntityRecognizer):
            def __init__(self) -> None:
                super().__init__(
                    supported_entities=supported_entities
                    or [
                        "CCCD",
                        "CMND",
                        "PHONE_NUMBER",
                        "MST",
                        "BANK_CARD",
                        "BANK_ACCOUNT",
                        "PASSPORT",
                        "VEHICLE_PLATE",
                    ],
                    name="VipiiRecognizer",
                )
                self.detector = PIIDetector()

            def load(self) -> None:
                return None

            def analyze(self, text, entities, nlp_artifacts=None):  # type: ignore[no-untyped-def]
                from presidio_analyzer import RecognizerResult

                allowed = set(entities)
                return [
                    RecognizerResult(
                        entity_type=match.label,
                        start=match.start,
                        end=match.end,
                        score=match.score,
                    )
                    for match in self.detector.detect(text)
                    if match.label in allowed
                ]

        self.recognizer = _Recognizer()

    def unwrap(self):  # type: ignore[no-untyped-def]
        return self.recognizer
