"""Recognizer interfaces and implementations."""

from __future__ import annotations

from vipii.recognizers.base import Recognizer
from vipii.recognizers.ner import NERRecognizer
from vipii.recognizers.pattern import PatternRecognizer, custom_pattern_recognizer
from vipii.recognizers.registry import RecognizerRegistry, built_in_recognizers
from vipii.recognizers.validators import (
    Validator,
    valid_bank_account,
    valid_bank_card,
    valid_cccd,
    valid_cmnd,
    valid_date_of_birth,
    valid_device_id,
    valid_driver_license,
    valid_email,
    valid_health_insurance,
    valid_ip_address,
    valid_passport,
    valid_phone,
    valid_social_insurance,
    valid_tax_code,
    valid_vehicle_plate,
    validator_by_name,
)

__all__ = [
    "NERRecognizer",
    "PatternRecognizer",
    "Recognizer",
    "RecognizerRegistry",
    "Validator",
    "built_in_recognizers",
    "custom_pattern_recognizer",
    "valid_bank_account",
    "valid_bank_card",
    "valid_cccd",
    "valid_cmnd",
    "valid_date_of_birth",
    "valid_device_id",
    "valid_driver_license",
    "valid_email",
    "valid_health_insurance",
    "valid_ip_address",
    "valid_passport",
    "valid_phone",
    "valid_social_insurance",
    "valid_tax_code",
    "valid_vehicle_plate",
    "validator_by_name",
]
