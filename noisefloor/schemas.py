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


def described(schema, description, **annotations):
    """Add discovery guidance without changing a shared validation contract."""
    return {**schema, "description": description, **annotations}


NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
PROBABILITY = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
CLOCK = dict(string(64), format="utc-date-time", description=(
    "UTC ISO 8601 timestamp, for example 2026-01-01T12:00:00Z. "
    "Use Z or +00:00 and at most six fractional second digits."))
NUMBERS = array(NUMBER, MAX_SERIES_POINTS, 1)
OBSERVATION = obj({
    "observed_at": described(CLOCK, "Time represented by this observation, in UTC. Timestamps within a series must be unique and strictly increasing."),
    "available_at": described(CLOCK, "UTC time the observation became available to the caller. A time before observed_at becomes an availability_precedes_observation visibility gap; after as_of becomes not_yet_available. Omission is retained as incomplete availability history."),
    "value": {"type": ["number", "null"], "description": "Finite measurement in the declared series unit. Prices must be positive and counts nonnegative. Use null for missing data; it breaks the contiguous analysis window and is never filled."},
    "sample_size": {"type": ["number", "null"], "exclusiveMinimum": 0, "description": "Positive measurement-coverage denominator for this observation, or null when missing. This is not trading volume. Omit if coverage is not supplied; zero is invalid."},
}, ("observed_at", "value"))
MARKET_SOURCE = obj({
    "id": described(string(), "Caller-defined source identifier retained in results, for example synthetic-demo or a licensed provider name."),
    "url": described(string(2048), "Optional source or evidence URL retained for attribution; no URL is fetched or independently verified."),
    "rights": {"type": "string", "enum": ["permitted", "unknown", "restricted"], "description": "Caller's source-use declaration. Only permitted permits numerical assessment; unknown and restricted remain explicit visibility gaps. The server does not grant or verify rights."},
}, ("id", "rights"))
MARKET_SERIES = obj({
    "id": described(string(), "Unique series identifier within this request, used in results and cross-series relationships."),
    "kind": {"type": "string", "enum": ["price", "return", "level", "spread", "rate", "count"], "description": "Measurement type. Both assessment tools convert price to log returns. market_assessment keeps other values unchanged; spectral_assessment keeps return values and first-differences level, spread, rate and count."},
    "unit": described(string(64), "Consistent measurement unit for all observations, for example USD, decimal_return, basis_points or count. No currency conversion, percent conversion or economic comparability is inferred."),
    "source": described(MARKET_SOURCE, "Source identity, optional attribution URL and declared usage rights. All are caller-supplied, not attestations."),
    "max_age_seconds": described(POSITIVE, "Maximum allowed age in seconds of the last observation relative to as_of. Older data remains insufficient_visibility; choose an appropriate cadence-specific limit."),
    "max_gap_seconds": described(POSITIVE, "Optional largest allowed time gap in seconds between observations. A larger gap restarts the contiguous tail. If omitted, calendar gap policy is unknown and a warning is returned."),
    "group": described(string(), "Optional caller-defined grouping label for review queues; defaults to this series id. It does not certify independence or comparability."),
    "observations": described(array(OBSERVATION, MAX_SERIES_POINTS), "Zero to 2048 observations in strictly increasing UTC order. Empty or inadequate histories return visibility gaps; null values and declared time gaps are preserved, never interpolated."),
}, ("id", "kind", "unit", "source", "max_age_seconds", "observations"))
MARKET_POLICY = obj({
    "baseline_points": {"type": "integer", "minimum": 10, "maximum": 256, "default": 30, "description": "Number of earlier transformed observations in the baseline window; default 30. Total required transformed history is baseline_points + recent_points."},
    "recent_points": {"type": "integer", "minimum": 1, "maximum": 32, "default": 3, "description": "Number of latest transformed observations compared with the baseline; default 3. Persistence requires at least two qualifying readings, so a one-point window cannot establish persistence."},
    "move_threshold": described(POSITIVE, "Departure threshold in baseline robust-spread units (1.4826 times median absolute deviation); default 3.0. This is a review threshold, not a significance level.", default=3.0),
    "volatility_ratio": {"type": "number", "exclusiveMinimum": 1, "default": 3.0, "description": "Recent-to-baseline robust-spread ratio that flags volatility expansion; default 3.0. A constant baseline followed by positive spread is also flagged."},
    "coverage_drop_fraction": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2, "description": "Absolute fractional change in median measurement coverage that prompts review, including increases; default 0.2 means 20%. Applies only to supplied sample_size values."},
})
MARKET_REQUEST = obj({
    "series": described(array(MARKET_SERIES, MAX_METRICS), "Up to 32 caller-supplied series to assess independently, with unique ids and source/clock metadata. An empty list returns insufficient_visibility."),
    "as_of": described(CLOCK, "UTC decision-time cutoff for source freshness and availability checks; for example 2026-01-01T12:00:00Z. Future observations are not silently included."),
    "policy": described(MARKET_POLICY, "Optional review thresholds; omitted fields use the documented defaults. Policies do not create statistical or trading guarantees."),
}, ("series", "as_of"))
SPECTRAL_POLICY = obj({
    "window_points": {"type": "integer", "minimum": 20, "maximum": 512, "default": 60, "description": "Aligned transformed intervals per correlation window; default 60. Must exceed the supplied series count; needs at least window_points + 1 original observations per series, including returns."},
    "step_points": {"type": "integer", "minimum": 1, "maximum": 512, "default": 20, "description": "Step back in aligned intervals between window endpoints; default 20. Overlapping windows are descriptive and not independent tests."},
    "max_windows": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4, "description": "Maximum recent complete windows returned, in chronological order; default 4. Fewer are returned when only a shorter eligible history is available."},
})
SPECTRAL_REQUEST = obj({
    "series": described(array(MARKET_SERIES, MAX_METRICS), "Caller-defined comparable panel, normally 2–32 series with unique ids. All members must pass source and clock gates. Both endpoints of each interval must align; rejected members are never silently removed."),
    "as_of": described(CLOCK, "UTC cutoff for freshness and availability, for example 2026-01-01T12:00:00Z. Caller timestamps do not certify historical as-published vintages."),
    "policy": described(SPECTRAL_POLICY, "Optional rolling-window sizes; omitted fields default to 60 intervals, 20-interval steps and at most 4 windows."),
}, ("series", "as_of"))
DYSON_REQUEST = obj({
    "dimension": {"type": "integer", "minimum": 2, "maximum": 16, "default": 6, "description": "Size N of the synthetic real-symmetric N-by-N matrix and number of eigenvalues per step; default 6."},
    "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 40, "description": "Number of Brownian increments; default 40. The returned path contains steps + 1 snapshots, including the initial zero matrix."},
    "dt": {"type": "number", "minimum": 0.000001, "maximum": 1, "default": 0.05, "description": "Synthetic time increment; default 0.05. This is not seconds or a market horizon. Off-diagonal increment variance is dt/N; diagonal variance is 2*dt/N."},
    "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 32 - 1, "default": 0, "description": "Integer random seed for deterministic replay in a recorded Python runtime; default 0. Record the implementation fingerprint and runtime with results."},
})
NARRATIVE_SOURCE = obj({
    "id": described(string(), "Caller-defined source identifier retained with the headline."),
    "url": described(string(2048), "Optional attribution URL, never fetched. Its hostname is the source-family fallback when family is omitted."),
    "kind": {"type": "string", "enum": ["primary", "reporting", "commentary"], "description": "Caller-declared source type, prioritized in this order: primary, reporting, commentary. This label does not verify truth, authority or independence."},
    "family": described(string(), "Optional shared source-family label for syndication or common ownership. Defaults to URL hostname, then source id. Distinct families are not verified independent corroboration."),
}, ("id", "kind"))
EVENT = obj({
    "id": described(string(), "Unique event identifier within the request, retained in event records and cluster member_ids."),
    "title": described(string(2000), "Headline text, up to 2000 characters, treated only as untrusted data. Wording, numbers and negation guide grouping; this is not full article analysis."),
    "published_at": described(CLOCK, "Headline publication time in UTC; future or stale headlines are retained with an exclusion reason."),
    "available_at": described(CLOCK, "Optional UTC time the headline became available to the caller; must not precede published_at. A value after as_of excludes it as not_yet_available. Omission prevents a verified as-published history claim."),
    "entities": described(array(string(), 64), "Up to 64 caller-assigned entity ids, for example SYNTHETIC_CO. Matching is exact and case-sensitive. An empty list remains unmapped; entities are not inferred from title text."),
    "source": described(NARRATIVE_SOURCE, "Caller-supplied source identity and type, with optional attribution and family. No retrieval or independent source verification occurs."),
    "event_key": described(string(), "Optional caller-defined event identity. Different explicit keys prevent merging; a shared key with different wording or quantities can flag separate conflict candidates for review."),
}, ("id", "title", "published_at", "entities", "source"))
NARRATIVE_POLICY = obj({
    "max_age_seconds": described(POSITIVE, "Maximum headline age in seconds at as_of; default 86400 (24 hours). Older events are retained as stale and excluded from candidate clusters.", default=86400),
    "max_items": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20, "description": "Maximum clusters in attention_queue; default 20. All clusters and events remain available, with overflow listed in deferred_cluster_ids."},
    "similarity_threshold": {"type": "number", "minimum": 0.5, "maximum": 1, "default": 0.8, "description": "Minimum token-set Jaccard similarity for grouping; default 0.8. Matching entities and compatible quantities, negation, direction and units are also required."},
})
NARRATIVE_REQUEST = obj({
    "events": described(array(EVENT, MAX_EVENTS), "Up to 500 caller-supplied headline records with unique ids. Supply the collection you are permitted to use; input order does not determine queue order. Empty input returns insufficient_visibility."),
    "as_of": described(CLOCK, "UTC cutoff for headline recency and availability, for example 2026-01-01T12:00:00Z."),
    "focus": described(array(string(), 64), "Optional entity ids of interest, matched exactly to each event's entities. Omitted or empty means no entity filter. Events with no mapped entity are retained for review; mapped off-focus events are excluded."),
    "policy": described(NARRATIVE_POLICY, "Optional triage settings; omitted fields use 86400 seconds maximum age, 20 queue items and 0.8 wording similarity."),
}, ("events", "as_of"))
RESULT_SCHEMA = {"type": "object", "additionalProperties": True,
                 "description": "Tool-specific result object described in each tool's description. Check status/ok/state, reading, validity and limits before using numbers. MCP returns this object in structuredContent and JSON text; invalid input returns isError=true with an error field. No tool fetches sources or executes trades."}


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
