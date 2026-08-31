"""Offline JSON Schema validation for dynamic offering and BPMN task contracts."""
import re
from datetime import date, time

from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable


def is_iso_date(value) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def is_local_time(value) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value):
        return False
    try:
        time.fromisoformat(value)
        return True
    except ValueError:
        return False


FORMATS = FormatChecker()
FORMATS.checks("date")(lambda value: not isinstance(value, str) or is_iso_date(value))
# The hotel contract uses local ISO times, not RFC 3339 instants with an offset.
FORMATS.checks("time")(lambda value: not isinstance(value, str) or is_local_time(value))


def satisfies_schema(value, schema) -> bool:
    if not isinstance(schema, (dict, bool)):
        return False
    try:
        validator = (validator_for(schema, default=None)
                     if isinstance(schema, dict) and "$schema" in schema else validator_for(schema))
        if validator is None:
            return False
        validator.check_schema(schema)
        # No remote schema retrieval, including for untrusted $ref values.
        return validator(schema, format_checker=FORMATS, registry=Registry()).is_valid(value)
    except (SchemaError, Unresolvable):
        return False
