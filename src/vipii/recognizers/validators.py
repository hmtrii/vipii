"""Validation helpers for Vietnamese structured PII recognizers."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

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


def valid_email(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            value.strip(),
            flags=re.IGNORECASE,
        )
    )


def valid_date_of_birth(value: str) -> bool:
    parts = re.split(r"[/.-]", value)
    if len(parts) != 3:
        return False
    day, month, year = (int(part) for part in parts)
    try:
        parsed = date(year, month, day)
    except ValueError:
        return False

    today = date.today()
    return date(today.year - 130, today.month, today.day) <= parsed <= today


def valid_tax_code(value: str) -> bool:
    digits = only_digits(value)
    return len(digits) in {10, 13}


def valid_bank_card(value: str) -> bool:
    digits = only_digits(value)
    return 16 <= len(digits) <= 19 and digits.startswith(("9704", "4", "5"))


def valid_bank_account(value: str) -> bool:
    digits = only_digits(value)
    return 8 <= len(digits) <= 16


def valid_social_insurance(value: str) -> bool:
    return len(only_digits(value)) == 10


def valid_health_insurance(value: str) -> bool:
    compact = re.sub(r"[\s.-]", "", value).upper()
    return bool(re.fullmatch(r"[A-Z]{2}\d{13}|\d{10}", compact))


def valid_passport(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][0-9]{7,8}", value.replace(" ", "").upper()))


def valid_vehicle_plate(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    return compact[:2] in VEHICLE_PLATE_PREFIXES and bool(
        re.fullmatch(r"\d{2}[A-Z][A-Z0-9]?[-.]?\d{3,5}(\.\d{2})?", compact)
    )


def valid_driver_license(value: str) -> bool:
    return len(only_digits(value)) == 12


def valid_ip_address(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def valid_device_id(value: str) -> bool:
    compact = value.strip()
    digits = only_digits(compact)
    return bool(
        re.fullmatch(
            r"[0-9A-F]{8}-[0-9A-F]{4}-[1-5][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}",
            compact,
            re.IGNORECASE,
        )
        or re.fullmatch(r"[0-9A-F]{2}(?::[0-9A-F]{2}){5}", compact, re.IGNORECASE)
        or len(digits) == 15
    )


VALIDATORS: dict[str, Validator] = {
    "cccd": valid_cccd,
    "cmnd": valid_cmnd,
    "phone": valid_phone,
    "phone_number": valid_phone,
    "email": valid_email,
    "email_address": valid_email,
    "date_of_birth": valid_date_of_birth,
    "dob": valid_date_of_birth,
    "tax_code": valid_tax_code,
    "mst": valid_tax_code,
    "bank_card": valid_bank_card,
    "bank_account": valid_bank_account,
    "social_insurance": valid_social_insurance,
    "bhxh": valid_social_insurance,
    "health_insurance": valid_health_insurance,
    "bhyt": valid_health_insurance,
    "passport": valid_passport,
    "vehicle_plate": valid_vehicle_plate,
    "driver_license": valid_driver_license,
    "ip_address": valid_ip_address,
    "device_id": valid_device_id,
}


def validator_by_name(name: str) -> Validator:
    try:
        return VALIDATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown validator '{name}'") from exc
