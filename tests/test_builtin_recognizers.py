from __future__ import annotations

import pkgutil

import pytest

from vipii import PIIDetector
from vipii.recognizers import PatternRecognizer
from vipii.recognizers import builtin as builtin_package
from vipii.recognizers.builtin import BUILTIN_MODULES, built_in_recognizers

EXPECTED_LABELS = {
    "CCCD",
    "CMND",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_OF_BIRTH",
    "MST",
    "SOCIAL_INSURANCE_NUMBER",
    "HEALTH_INSURANCE_NUMBER",
    "BANK_CARD",
    "BANK_ACCOUNT",
    "PASSPORT",
    "VEHICLE_PLATE",
    "DRIVER_LICENSE",
    "IP_ADDRESS",
    "DEVICE_ID",
}


def module_names() -> set[str]:
    return {
        info.name
        for info in pkgutil.iter_modules(builtin_package.__path__)
        if not info.name.startswith("_")
    }


def test_every_builtin_module_is_registered() -> None:
    registered = {module.__name__.rsplit(".", 1)[-1] for module in BUILTIN_MODULES}

    assert registered == module_names()


@pytest.mark.parametrize("module", BUILTIN_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_module_builds_a_recognizer_named_after_itself(module) -> None:  # type: ignore[no-untyped-def]
    recognizer = module.recognizer()

    assert isinstance(recognizer, PatternRecognizer)
    assert recognizer.name == module.__name__.rsplit(".", 1)[-1]
    assert recognizer.patterns
    assert recognizer.validator is not None
    assert recognizer.supported_entities == [recognizer.label]


def test_builtins_cover_the_expected_labels() -> None:
    assert {r.label for r in built_in_recognizers()} == EXPECTED_LABELS


def test_recognizer_names_are_unique() -> None:
    names = [r.name for r in built_in_recognizers()]

    assert len(names) == len(set(names))


def test_built_in_recognizers_returns_fresh_instances() -> None:
    first, second = built_in_recognizers(), built_in_recognizers()

    assert [r.name for r in first] == [r.name for r in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_detector_loads_builtins_from_modules() -> None:
    assert {r.name for r in PIIDetector().recognizers} == module_names()
