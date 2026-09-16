"""Built-in Vietnamese PII recognizers, one module per entity."""

from __future__ import annotations

from vipii.recognizers.builtin import (
    bank_account,
    bank_card,
    cccd,
    cmnd,
    date_of_birth,
    device_id,
    driver_license,
    email_address,
    health_insurance,
    ip_address,
    mst,
    passport,
    phone_number,
    social_insurance,
    vehicle_plate,
)
from vipii.recognizers.pattern import PatternRecognizer

# Registration is this tuple: add a module above, then list it here.
BUILTIN_MODULES = (
    cccd,
    cmnd,
    phone_number,
    email_address,
    date_of_birth,
    mst,
    social_insurance,
    health_insurance,
    bank_card,
    bank_account,
    passport,
    vehicle_plate,
    driver_license,
    ip_address,
    device_id,
)


def built_in_recognizers() -> list[PatternRecognizer]:
    """Build a fresh instance of every built-in recognizer."""
    return [module.recognizer() for module in BUILTIN_MODULES]


__all__ = ["BUILTIN_MODULES", "built_in_recognizers"]
