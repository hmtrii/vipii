"""Validation helpers for Vietnamese structured PII recognizers."""

from __future__ import annotations

import re
from collections.abc import Callable

from vipii.constants import CCCD_PROVINCE_CODES, MOBILE_PREFIXES, VEHICLE_PLATE_PREFIXES

Validator = Callable[[str], bool]


def only_digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def valid_cccd(value: str) -> bool:
    digits = only_digits(value)
    return len(digits) == 12 and digits[:3] in CCCD_PROVINCE_CODES and digits[3].isdigit()


def valid_cmnd(value: str) -> bool:
    return len(only_digits(value)) == 9


def valid_phone(value: str) -> bool:
    digits = only_digits(value)
    if digits.startswith("84"):
        digits = "0" + digits[2:]
    return len(digits) == 10 and digits[:3] in MOBILE_PREFIXES


def valid_tax_code(value: str) -> bool:
    digits = only_digits(value)
    return len(digits) in {10, 13}


def valid_bank_card(value: str) -> bool:
    digits = only_digits(value)
    return 16 <= len(digits) <= 19 and digits.startswith(("9704", "4", "5"))


def valid_bank_account(value: str) -> bool:
    digits = only_digits(value)
    return 8 <= len(digits) <= 16


def valid_passport(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][0-9]{7,8}", value.replace(" ", "").upper()))


def valid_vehicle_plate(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    return compact[:2] in VEHICLE_PLATE_PREFIXES and bool(
        re.fullmatch(r"\d{2}[A-Z][A-Z0-9]?[-.]?\d{3,5}(\.\d{2})?", compact)
    )


VALIDATORS: dict[str, Validator] = {
    "cccd": valid_cccd,
    "cmnd": valid_cmnd,
    "phone": valid_phone,
    "phone_number": valid_phone,
    "tax_code": valid_tax_code,
    "mst": valid_tax_code,
    "bank_card": valid_bank_card,
    "bank_account": valid_bank_account,
    "passport": valid_passport,
    "vehicle_plate": valid_vehicle_plate,
}


def validator_by_name(name: str) -> Validator:
    try:
        return VALIDATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown validator '{name}'") from exc
