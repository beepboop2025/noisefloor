"""Bounded JSON contracts shared by MCP, REST and the offline CLI.

The validator deliberately implements only keywords used by these owned schemas;
it is not a general JSON Schema implementation.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_SERIES_POINTS = 2048
MAX_METRICS = 32
MAX_BATCH = 16
MAX_EVENTS = 500


def obj(properties, required=(), **kwargs):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False, **kwargs}


def array(items, maximum, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def string(maximum=128, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
PROBABILITY = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
CLOCK = dict(string(64), format="utc-date-time")
NUMBERS = array(NUMBER, MAX_SERIES_POINTS, 1)
OBSERVATION = obj({
    "observed_at": CLOCK,
    "available_at": CLOCK,
    "value": {"type": ["number", "null"]},
    "sample_size": {"type": ["number", "null"], "exclusiveMinimum": 0},
}, ("observed_at", "value"))
MARKET_SOURCE = obj({
    "id": string(), "url": string(2048),
    "rights": {"type": "string", "enum": ["permitted", "unknown", "restricted"]},
}, ("id", "rights"))
MARKET_SERIES = obj({
    "id": string(), "kind": {"type": "string", "enum": ["price", "return", "level", "spread", "rate", "count"]},
    "unit": string(64), "source": MARKET_SOURCE,
    "max_age_seconds": POSITIVE, "max_gap_seconds": POSITIVE,
    "group": string(), "observations": array(OBSERVATION, MAX_SERIES_POINTS),
}, ("id", "kind", "unit", "source", "max_age_seconds", "observations"))
MARKET_POLICY = obj({
    "baseline_points": {"type": "integer", "minimum": 10, "maximum": 256},
    "recent_points": {"type": "integer", "minimum": 1, "maximum": 32},
    "move_threshold": POSITIVE,
    "volatility_ratio": {"type": "number", "exclusiveMinimum": 1},
    "coverage_drop_fraction": {"type": "number", "minimum": 0, "maximum": 1},
})
MARKET_REQUEST = obj({
    "series": array(MARKET_SERIES, MAX_METRICS), "as_of": CLOCK, "policy": MARKET_POLICY,
}, ("series", "as_of"))
SPECTRAL_POLICY = obj({
    "window_points": {"type": "integer", "minimum": 20, "maximum": 512},
    "step_points": {"type": "integer", "minimum": 1, "maximum": 512},
    "max_windows": {"type": "integer", "minimum": 1, "maximum": 8},
})
SPECTRAL_REQUEST = obj({
    "series": array(MARKET_SERIES, MAX_METRICS), "as_of": CLOCK,
    "policy": SPECTRAL_POLICY,
}, ("series", "as_of"))
DYSON_REQUEST = obj({
    "dimension": {"type": "integer", "minimum": 2, "maximum": 16},
    "steps": {"type": "integer", "minimum": 1, "maximum": 100},
    "dt": {"type": "number", "minimum": 0.000001, "maximum": 1},
    "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 32 - 1},
})
NARRATIVE_SOURCE = obj({
    "id": string(), "url": string(2048),
    "kind": {"type": "string", "enum": ["primary", "reporting", "commentary"]},
    "family": string(),
}, ("id", "kind"))
EVENT = obj({
    "id": string(), "title": string(2000), "published_at": CLOCK, "available_at": CLOCK,
    "entities": array(string(), 64), "source": NARRATIVE_SOURCE, "event_key": string(),
}, ("id", "title", "published_at", "entities", "source"))
NARRATIVE_POLICY = obj({
    "max_age_seconds": POSITIVE,
    "max_items": {"type": "integer", "minimum": 1, "maximum": 100},
    "similarity_threshold": {"type": "number", "minimum": 0.5, "maximum": 1},
})
NARRATIVE_REQUEST = obj({
    "events": array(EVENT, MAX_EVENTS), "as_of": CLOCK,
    "focus": array(string(), 64), "policy": NARRATIVE_POLICY,
}, ("events", "as_of"))
RESULT_SCHEMA = {"type": "object", "additionalProperties": True}


def parse_utc(value):
    """Parse the same microsecond-resolution UTC contract on Python 3.9+."""
    if not isinstance(value, str):
        raise ValueError("UTC ISO timestamp required")
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(?:Z|\+00:00)", value)
    if not match:
        raise ValueError("UTC ISO timestamp required, with at most six fractional digits")
    fraction = "." + match[2].ljust(6, "0") if match[2] else ""
    return datetime.fromisoformat(match[1] + fraction + "+00:00")


def _kind(value, name):
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": lambda: value is None,
    }[name]()


def validate(value, schema, path="input"):
    """Validate the explicitly supported JSON Schema subset without coercion."""
    kinds = schema.get("type", [])
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_kind(value, kind) for kind in kinds):
        raise ValueError(f"{path}: expected {' or '.join(kinds)}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in the allowed set")
    if value is None:
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{path}: finite number required")
        for key, invalid in (
            ("minimum", lambda n: value < n), ("maximum", lambda n: value > n),
            ("exclusiveMinimum", lambda n: value <= n), ("exclusiveMaximum", lambda n: value >= n),
        ):
            if key in schema and invalid(schema[key]):
                raise ValueError(f"{path}: violates {key} {schema[key]}")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 65536):
            raise ValueError(f"{path}: invalid string length")
        if schema.get("format") == "utc-date-time":
            try:
                parse_utc(value)
            except ValueError:
                raise ValueError(f"{path}: UTC ISO timestamp required") from None
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", MAX_SERIES_POINTS):
            raise ValueError(f"{path}: invalid array length")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, dict):
        if not schema.get("minProperties", 0) <= len(value) <= schema.get("maxProperties", 128):
            raise ValueError(f"{path}: invalid object size")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key}: required")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError(f"{path}: object keys must be bounded strings")
            if key in props:
                validate(item, props[key], f"{path}.{key}")
            else:
                additional = schema.get("additionalProperties", True)
                if additional is False:
                    raise ValueError(f"{path}: unknown field {key}")
                if isinstance(additional, dict):
                    validate(item, additional, f"{path}.{key}")


def loads(data):
    """Read strict, bounded JSON: reject duplicate keys and nonfinite literals."""
    if len(data.encode("utf-8") if isinstance(data, str) else data) > MAX_BODY_BYTES:
        raise ValueError("request body exceeds the byte limit")

    def constant(_value):
        raise ValueError("nonfinite JSON number")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(data, parse_constant=constant, object_pairs_hook=pairs)
    except (UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("invalid JSON encoding or nesting") from exc


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
