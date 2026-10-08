"""Offline, explicit adapters into the market-series contract.

These functions transform caller-supplied records. They fetch nothing, infer no
licence, and do not decide which revision is correct. The returned envelope
keeps provenance separate from the strict numerical input to ``market.assess``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import re

from .schemas import CLOCK, MARKET_SERIES, MAX_BODY_BYTES, MAX_METRICS, MAX_SERIES_POINTS, validate, parse_utc


_FIELDS = {
    "timestamp": "timestamp", "value": "value", "available_at": "available_at",
    "sample_size": "sample_size", "unit": "unit", "source": "source", "rights": "rights",
}
_RIGHTS = {"permitted", "unknown", "restricted"}
# Keep the native Financial Evidence raw-observation exclusions. Source state,
# rights and transport must each pass; one positive declaration cannot override
# a restriction in another field.
_BLOCKED_STATES = (
    "restricted", "withheld", "unavailable", "not_available", "denied",
    "pending", "review_required", "prohibited", "metadata_only", "derived_only",
    "review_hold", "blocked", "stale", "not_permitted",
)
_KNOWN_SOURCE_STATES = {"fresh", "observed", "published", "available", "complete",
                        "red", "orange", "yellow", "green"}


def _state(value):
    return re.sub(r"[\s-]+", "_", value.strip().lower()) if isinstance(value, str) else "unknown"


def _blocked_state(value):
    return any(marker in _state(value) for marker in _BLOCKED_STATES)


def _copy_json(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("records must contain finite, JSON-compatible data") from exc
    if len(encoded.encode("utf-8")) > MAX_BODY_BYTES:
        raise ValueError("adapter input exceeds the byte limit")
    return deepcopy(value)


def _clock(value, label, *, allow_date_only, issues, index):
    if isinstance(value, str) and len(value) == 10:
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{label}: invalid date") from None
        if not allow_date_only:
            raise ValueError(f"{label}: date-only input requires allow_date_only=True")
        issues.append({"record_index": index, "code": "date_only_normalized",
                       "field": label, "original": value,
                       "detail": "00:00 UTC is an explicit day label, not an observed intraday time."})
        value += "T00:00:00Z"
    validate(value, CLOCK, label)
    return parse_utc(value).isoformat().replace("+00:00", "Z")


def _rights(value):
    # Only an explicit permitted status confers permission. Public reachability,
    # a URL, and an HTTP 200 are deliberately absent from this mapping.
    state = _state(value)
    if state in _RIGHTS:
        return state
    if state == "allowed":
        return "permitted"
    if _blocked_state(value):
        return "restricted"
    return "unknown"


def _least_rights(values):
    statuses = {_rights(value) for value in values}
    return "restricted" if "restricted" in statuses else "unknown" if "unknown" in statuses else "permitted"


def from_records(records, *, series_id, kind, unit, source, max_age_seconds,
                 mapping=None, identity_fields=("entity_id", "metric", "symbol"),
                 allow_date_only=False, max_gap_seconds=None, group=None):
    """Adapt one explicitly identified series, refusing mixed identities.

    Default columns: timestamp, value, available_at, sample_size, unit, source,
    rights. ``mapping`` maps these canonical names to caller column names.
    ``source`` uses the market contract {id, rights, url?}; rights is a caller
    declaration, never a licence assessment. Missing numeric values remain null.
    Duplicate observation clocks require explicit revision selection upstream.
    """
    records = _copy_json(records)
    if not isinstance(records, list) or len(records) > MAX_SERIES_POINTS:
        raise ValueError(f"records must be a list of at most {MAX_SERIES_POINTS} entries")
    if not isinstance(allow_date_only, bool):
        raise ValueError("allow_date_only must be boolean")
    if not isinstance(identity_fields, (list, tuple)) or any(
        not isinstance(key, str) or not key for key in identity_fields
    ):
        raise ValueError("identity_fields must contain column names")
    columns = dict(_FIELDS)
    if mapping is not None:
        if not isinstance(mapping, dict) or set(mapping) - set(columns):
            raise ValueError("mapping contains unknown canonical fields")
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError("mapping values must be column names")
        columns.update(mapping)
    if len(set(columns.values())) != len(columns):
        raise ValueError("mapping must not map different meanings to one column")
    source = _copy_json(source)
    series = {"id": series_id, "kind": kind, "unit": unit, "source": deepcopy(source),
              "max_age_seconds": max_age_seconds, "observations": []}
    if max_gap_seconds is not None:
        series["max_gap_seconds"] = max_gap_seconds
    if group is not None:
        series["group"] = group
    validate(series, MARKET_SERIES, "series")
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("every record must be an object")
    for key in identity_fields:
        identities = {json.dumps(row.get(key), sort_keys=True) for row in records}
        if len(identities) > 1:
            raise ValueError(f"mixed {key}: partition institutions, instruments and metrics before adapting")

    issues, seen, rights = [], set(), [source["rights"]]
    for index, row in enumerate(records):
        if columns["unit"] in row and row[columns["unit"]] != unit:
            raise ValueError(f"records[{index}]: unit conflicts with the declared unit")
        if columns["source"] in row:
            declared = row[columns["source"]]
            if isinstance(declared, str):
                if declared != source["id"]:
                    raise ValueError(f"records[{index}]: mixed sources")
            elif isinstance(declared, dict):
                if set(declared) - {"id", "url", "rights"} or declared.get("id") != source["id"]:
                    raise ValueError(f"records[{index}]: mixed or unrecognized sources")
                if "url" in declared and declared["url"] != source.get("url"):
                    raise ValueError(f"records[{index}]: source URL conflicts with declared source")
                rights.append(declared.get("rights", "unknown"))
            else:
                raise ValueError(f"records[{index}]: source must be an id or source object")
        if columns["rights"] in row:
            rights.append(row[columns["rights"]])
        observed = _clock(row.get(columns["timestamp"]), f"records[{index}].timestamp",
                          allow_date_only=allow_date_only, issues=issues, index=index)
        if observed in seen:
            raise ValueError(f"records[{index}]: duplicate observation timestamp; select revisions explicitly")
        seen.add(observed)
        point = {"observed_at": observed, "value": row.get(columns["value"])}
        if columns["value"] not in row:
            issues.append({"record_index": index, "code": "missing_value_preserved_as_null"})
        if row.get(columns["available_at"]) is not None:
            point["available_at"] = _clock(
                row[columns["available_at"]], f"records[{index}].available_at",
                allow_date_only=False, issues=issues, index=index)
        if columns["sample_size"] in row:
            point["sample_size"] = row[columns["sample_size"]]
        series["observations"].append(point)
    series["source"]["rights"] = _least_rights(rights)
    if series["source"]["rights"] != source["rights"]:
        issues.append({"code": "rights_downgraded", "detail": "The least permissive explicit row/source status wins."})
    before = [point["observed_at"] for point in series["observations"]]
    series["observations"].sort(key=lambda point: parse_utc(point["observed_at"]))
    if before != [point["observed_at"] for point in series["observations"]]:
        issues.append({"code": "sorted_by_observation_time"})
    validate(series, MARKET_SERIES, "series")
    return {"schema": "noisefloor.adapter.v1", "series": [series], "issues": issues,
            "provenance": {"adapter": "explicit_records", "mapping": columns,
                           "identity_fields": list(identity_fields), "records": records,
                           "source_declaration": _copy_json(source)},
            "scope": "caller_records_only; no network, inferred licence or revision selection"}


def from_financial_evidence(table, *, kind, max_age_seconds, allow_date_only=False,
                            max_gap_seconds=None):
    """Adapt Financial Evidence table rows without importing its runtime.

    Partition by product, dataset, institution, metric, unit and source URL.
    Undertow liquidity measures additionally require their native entity_name;
    entity_id alone identifies the segment, not the individual measurement.
    Observation clocks come only from ``as_of``. Actual availability comes only
    from ``knowledge_time``; ``published_at`` and ``retrieved_at`` remain distinct
    in provenance. Missing clocks or identities fail instead of disappearing.
    """
    table = _copy_json(table)
    if not isinstance(table, dict) or table.get("schema") not in {
        "liquidity-lab.openbb-table.v1", "financial-evidence.agent-result.v1",
    }:
        raise ValueError("expected a supported Financial Evidence table schema")
    rows = table.get("results")
    if not isinstance(rows, list) or len(rows) > MAX_METRICS * MAX_SERIES_POINTS:
        raise ValueError("Financial Evidence results must be a bounded list")
    if (any(type(table.get(key)) is not int for key in ("offset", "returned_rows", "total_rows"))
            or "next_offset" not in table or table["next_offset"] is not None
            or table.get("offset") != 0 or table.get("returned_rows") != len(rows)
            or table.get("total_rows") != len(rows)):
        raise ValueError("table is incomplete: collect the intended complete page range explicitly")
    groups = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"results[{index}] must be an object")
        names = ("product", "dataset", "entity_id", "metric", "unit", "source_url")
        if any(not isinstance(row.get(name), str) or not row[name] for name in names):
            raise ValueError(f"results[{index}] lacks an explicit product/dataset/entity/metric/unit/source identity")
        measure = ""
        if row["dataset"] == "market_liquidity":
            measure = row.get("entity_name")
            if not isinstance(measure, str) or not measure:
                raise ValueError(f"results[{index}] lacks an explicit liquidity measure identity")
        identity = tuple(row[name] for name in names) + (measure,)
        groups.setdefault(identity, []).append((index, row))
    if len(groups) > MAX_METRICS:
        raise ValueError(f"Financial Evidence table contains more than {MAX_METRICS} distinct series")
    series, issues, identities = [], [], []
    for identity, entries in sorted(groups.items()):
        product, dataset, entity_id, metric, unit, source_url, measure = identity
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
        series_id = "fe:" + digest[:24]
        source_id = "fe-source:" + hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
        source = {"id": source_id, "url": source_url,
                  "rights": _least_rights([row.get("rights_status") for _, row in entries])}
        adapted = []
        for original_index, row in entries:
            value = row.get("value")
            block_reasons = []
            if _state(row.get("availability")) not in {"available", "published", "observed"}:
                block_reasons.append("availability_not_established")
            if _blocked_state(row.get("source_status")):
                block_reasons.append("source_status_restricted")
            elif _state(row.get("source_status")) not in _KNOWN_SOURCE_STATES:
                block_reasons.append("source_status_unknown")
            if _rights(row.get("rights_status")) != "permitted":
                block_reasons.append("source_rights_not_permitted")
            if _state(row.get("transport_status")) != "complete":
                block_reasons.append("row_transport_not_complete")
            if _state(table.get("transport_status")) != "complete":
                block_reasons.append("table_transport_not_complete")
            if row.get("publication_allowed") is False or table.get("publication_allowed") is False:
                block_reasons.append("publication_not_allowed")
            if block_reasons:
                value = None
                issues.append({"record_index": original_index, "series_id": series_id,
                               "code": "ineligible_value_withheld",
                               "reasons": block_reasons,
                               "detail": "Source availability/transport blocks were preserved; original row remains in provenance."})
            adapted.append({"timestamp": row.get("as_of"), "value": value,
                            "available_at": row.get("knowledge_time"),
                            "entity_id": entity_id, "metric": metric})
            if row.get("knowledge_time") is None:
                issues.append({"record_index": original_index, "series_id": series_id,
                               "code": "availability_unknown",
                               "detail": "Publication/retrieval times were not substituted for knowledge_time."})
        result = from_records(adapted, series_id=series_id, kind=kind, unit=unit,
                              source=source, max_age_seconds=max_age_seconds,
                              max_gap_seconds=max_gap_seconds, allow_date_only=allow_date_only,
                              group=entity_id)
        series.extend(result["series"])
        for issue in result["issues"]:
            issue = dict(issue, series_id=series_id)
            if "record_index" in issue:
                issue["record_index"] = entries[issue["record_index"]][0]
            issues.append(issue)
        identities.append({"series_id": series_id, "product": product, "dataset": dataset,
                           "entity_id": entity_id, "metric": metric, "unit": unit,
                           "measure": measure or None,
                           "source_url": source_url, "record_indices": [index for index, _ in entries]})
    return {"schema": "noisefloor.adapter.v1", "series": series, "issues": issues,
            "provenance": {"adapter": "financial_evidence_table", "identities": identities,
                           "input": table},
            "scope": "published observations only; not a verified as-published vintage or source-rights approval"}
