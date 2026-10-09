"""Offline market attention diagnostics, with explicit data and inference limits.

This module neither collects data nor recommends trades. A quiet diagnostic is
not a finding that information is noise. Every reading retains its source clock.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import median

from . import change
from .schemas import MARKET_REQUEST, validate, parse_utc

DEFAULT_POLICY = {
    "baseline_points": 30, "recent_points": 3, "move_threshold": 3.0,
    "volatility_ratio": 3.0, "coverage_drop_fraction": 0.2,
}


def _time(value):
    return parse_utc(value)


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _number(value):
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("derived statistic overflow; rescale the input units")
    # Fixed decimal rounding erases genuine movements in small-unit series.
    return float(format(value, ".15g"))


def _spread(values):
    center = median(values)
    return 1.4826 * median([abs(x - center) for x in values])


def _correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 10:
        return None
    def normalized_deviations(values):
        magnitude = max(abs(value) for value in values)
        if magnitude == 0:
            return None
        scaled = [value / magnitude for value in values]
        center = math.fsum(scaled) / len(scaled)
        deviations = [value - center for value in scaled]
        norm = math.sqrt(math.fsum(value * value for value in deviations))
        return [value / norm for value in deviations] if norm else None

    # Normalize before centering/squaring: finite 1e200 inputs otherwise have
    # infinite squared norms, while 1e-200 inputs can have zero squared norms.
    dx, dy = normalized_deviations(xs), normalized_deviations(ys)
    if dx is None or dy is None:
        return None
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(dx, dy))))


def _prepare_series(item, as_of):
    """Shared source gates and contiguous tail; never fill a missing observation."""
    obs = item["observations"]
    times = [_time(p["observed_at"]) for p in obs]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{item['id']}: observations must have unique increasing timestamps")
    if item["kind"] == "price" and any(p["value"] is not None and p["value"] <= 0 for p in obs):
        raise ValueError(f"{item['id']}: prices must be strictly positive")
    if item["kind"] == "count" and any(p["value"] is not None and p["value"] < 0 for p in obs):
        raise ValueError(f"{item['id']}: counts cannot be negative")
    issues, warnings = [], []
    if item["source"]["rights"] != "permitted":
        issues.append("source_use_not_permitted_or_unknown")
    if not obs:
        issues.append("no_observations")
    if any(t > as_of for t in times):
        issues.append("future_observation")
    if any("available_at" in p and _time(p["available_at"]) > as_of for p in obs):
        issues.append("not_yet_available")
    if any("available_at" in p and _time(p["available_at"]) < t for p, t in zip(obs, times)):
        issues.append("availability_precedes_observation")
    if any("available_at" not in p for p in obs):
        warnings.append("availability_history_incomplete")
    age = (as_of - times[-1]).total_seconds() if times else None
    if age is not None and age > item["max_age_seconds"]:
        issues.append("stale_observation")

    # Start again after a missing observation or a caller-declared calendar gap.
    # Never interpolate, forward-fill, or calculate returns across such a gap.
    start, gaps = 0, []
    for i, point in enumerate(obs):
        if point["value"] is None:
            start = i + 1
        if i and "max_gap_seconds" in item and (times[i] - times[i-1]).total_seconds() > item["max_gap_seconds"]:
            start = max(start, i)
            gaps.append(i)
    if "max_gap_seconds" not in item:
        warnings.append("calendar_gap_policy_not_supplied")
    if any(p["value"] is None for p in obs):
        warnings.append("missing_observations_preserved")
    if gaps:
        warnings.append("calendar_gaps_preserved")
    if obs and obs[-1]["value"] is None:
        issues.append("latest_value_missing")

    return {"observations": obs, "times": times, "tail": obs[start:],
            "issues": issues, "warnings": warnings, "age": age, "gaps": gaps}


def _inspect(item, as_of, policy):
    prepared = _prepare_series(item, as_of)
    obs, tail = prepared["observations"], prepared["tail"]
    issues, warnings = prepared["issues"], prepared["warnings"]
    age, gaps = prepared["age"], prepared["gaps"]
    derived = []
    representation_errors = []
    if item["kind"] == "price":
        for previous, current in zip(tail, tail[1:]):
            # Subtract logs instead of dividing to avoid over/underflow.
            before, after = math.log(previous["value"]), math.log(current["value"])
            delta = after - before
            derived.append((current["observed_at"], delta))
            # Bound arithmetic representation noise at the actual log-price
            # scale. This is not an economic/materiality threshold, and must
            # never be imposed on caller-supplied spreads, rates or levels.
            representation_errors.append(4 * (math.ulp(before) + math.ulp(after) + math.ulp(delta)))
        transform, analyzed_unit = "log_return", "log_return"
    else:
        derived = [(p["observed_at"], float(p["value"])) for p in tail]
        transform, analyzed_unit = "identity", item["unit"]
    need = policy["baseline_points"] + policy["recent_points"]
    if len(derived) < need:
        issues.append("insufficient_contiguous_history")
    result = {
        "id": item["id"], "kind": item["kind"], "unit": item["unit"],
        "source": dict(item["source"]), "group": item.get("group", item["id"]),
        "status": "insufficient_visibility", "reasons": issues, "warnings": warnings,
        "observed_at": obs[-1]["observed_at"] if obs else None,
        "available_at": obs[-1].get("available_at") if obs else None,
        "age_seconds": age, "observation_count": len(obs),
        "contiguous_points": len(derived), "required_points": need,
        "missing_points": sum(p["value"] is None for p in obs),
        "gap_indices": gaps, "transform": transform, "analyzed_unit": analyzed_unit,
        "diagnostics": None, "priority": None, "is_e_value": False,
        "validity": "descriptive", "guarantee": None,
        "availability_timestamps_complete": bool(obs) and all("available_at" in p for p in obs),
        "as_published_history_verified": False,
    }
    if issues:
        result["reading"] = "Insufficient visibility: " + ", ".join(issues)
        return result, {}

    window = derived[-need:]
    values = [v for _, v in window]
    b = policy["baseline_points"]
    baseline, recent = values[:b], values[b:]
    center, scale = median(baseline), _spread(baseline)
    precision = max(representation_errors[-need:], default=0.0)
    if scale <= precision:
        scale = 0.0
    effect, latest = median(recent) - center, recent[-1] - center
    threshold = policy["move_threshold"]
    def unusual(delta):
        return abs(delta) > precision and (abs(delta) >= threshold * scale if scale else True)
    # Multiplying two small nonzero effects can underflow to zero.
    same_direction = sum(unusual(x-center) and
                         ((x > center and effect > 0) or (x < center and effect < 0))
                         for x in recent)
    persistent = unusual(effect) and same_direction >= max(2, math.ceil(len(recent)*2/3))
    isolated = unusual(latest)
    recent_spread = _spread(recent)
    if recent_spread <= precision:
        recent_spread = 0.0
    vol_ratio = recent_spread / scale if scale else None
    volatility = (vol_ratio is not None and vol_ratio >= policy["volatility_ratio"])
    if scale == 0 and recent_spread > 0:
        volatility = True

    reasons = []
    if persistent:
        reasons.append("persistent_departure")
    elif isolated:
        reasons.append("latest_departure")
    if volatility:
        reasons.append("volatility_expansion")
    if scale == 0 and (abs(effect) > precision or abs(latest) > precision or recent_spread > precision):
        reasons.append("constant_baseline_changed")

    # Measurement coverage is caller-declared. Economic trading volume must be
    # supplied as its own series; it is not a measurement denominator.
    coverage = {"status": "not_supplied", "relative_change": None}
    selected_obs = tail[-need:]
    if any("sample_size" in p for p in selected_obs):
        if any(p.get("sample_size") is None for p in selected_obs):
            coverage["status"] = "incomplete"
            warnings.append("measurement_coverage_incomplete")
        else:
            before = median([p["sample_size"] for p in selected_obs[:b]])
            after = median([p["sample_size"] for p in selected_obs[b:]])
            relative = (after - before) / before
            coverage = {"status": "changed" if abs(relative) >= policy["coverage_drop_fraction"] and relative != 0 else "stable",
                        "relative_change": _number(relative)}
            if coverage["status"] == "changed":
                warnings.append("measurement_coverage_changed")
    if coverage["status"] in ("changed", "incomplete"):
        reasons.append("review_measurement_coverage")
    status = "review" if persistent or volatility else "watch" if reasons else "no_supported_departure"
    if "review_measurement_coverage" in reasons:
        status = "review"
    monitor_values = [center if abs(value-center) <= precision else value for value in values]
    detector = change.scan(monitor_values)
    # This raw detector statistic is expressly not passed to e-BH.
    diagnostics = {
        "baseline_median": _number(center), "recent_median": _number(median(recent)),
        "median_change": _number(effect), "latest_change": _number(latest),
        "baseline_robust_spread": _number(scale), "recent_robust_spread": _number(recent_spread),
        "robust_departure": _number(effect / scale) if scale else None,
        "latest_robust_departure": _number(latest / scale) if scale else None,
        "volatility_ratio": _number(vol_ratio), "persistence_points": same_direction,
        "representation_tolerance": _number(precision),
        "representation_tolerance_basis": "log-price ULP arithmetic only" if precision else "none for directly supplied values",
        "lag_one_correlation": _number(_correlation(values[:-1], values[1:])),
        "coverage": coverage,
        "monitor": {"state": detector["state"], "statistic": detector["evidence"],
                    "statistic_type": "shiryaev_roberts", "is_e_value": False,
                    "change_points": detector["change_points"], "guarantee": None},
        "window_start": window[0][0], "window_end": window[-1][0],
    }
    result.update(status=status, reasons=reasons, diagnostics=diagnostics,
                  analyzed_points=len(window), priority=2 if status == "review" else 1 if status == "watch" else 0,
                  reading=("Review observed departure or measurement coverage; establish its cause and materiality."
                           if status == "review" else "Watch the latest departure; persistence is not established."
                           if status == "watch" else "No supported departure under this policy; this does not establish noise or safety."))
    # Equivalent source timestamp spellings denote the same instant. Preserve
    # the original source strings above, but align internal windows in UTC.
    return result, {_time(stamp).astimezone(timezone.utc): value
                    for (stamp, _), value in zip(window, monitor_values)}


def assess(series: list[dict], *, as_of: str, policy: dict | None = None) -> dict:
    """Assess bounded caller-supplied series without network, state or execution.

    Sources, permissions and timestamps are declarations, not independently
    certified. Policy thresholds prioritize review; they are not p-values.
    """
    request = {"series": series, "as_of": as_of}
    if policy is not None:
        request["policy"] = policy
    validate(request, MARKET_REQUEST)
    ids = [s["id"] for s in series]
    if len(ids) != len(set(ids)):
        raise ValueError("series ids must be unique")
    applied = {**DEFAULT_POLICY, **(policy or {})}
    now = _time(as_of)
    results, windows = [], {}
    for item in series:
        result, window = _inspect(item, now, applied)
        results.append(result)
        if window:
            windows[item["id"]] = window
    relationships = []
    for i, first in enumerate(sorted(windows)):
        for second in sorted(windows)[i+1:]:
            shared = sorted(set(windows[first]) & set(windows[second]))
            corr = _correlation([windows[first][t] for t in shared], [windows[second][t] for t in shared])
            if corr is not None and abs(corr) >= 0.85:
                relationships.append({"series": [first, second], "correlation": _number(corr),
                                      "aligned_points": len(shared), "independent_confirmation": False})
    missing = [r["id"] for r in results if r["status"] == "insufficient_visibility"]
    queue = sorted((r for r in results if r["status"] in ("review", "watch")),
                   key=lambda r: (-r["priority"], r["id"]))
    return {
        "schema": "noisefloor.market-assessment.v1", "as_of": as_of,
        "status": "insufficient_visibility" if not series or len(missing) == len(series) else "partial" if missing else "assessed",
        "policy": applied, "series": results,
        "attention_queue": [{"id": r["id"], "status": r["status"], "reasons": r["reasons"], "group": r["group"]} for r in queue],
        "visibility_gaps": missing, "related_movements": relationships,
        "coverage": {"requested_series": len(series), "assessed_series": len(series)-len(missing),
                     "complete_for_supplied_series": bool(series) and not missing, "market_universe_complete": False},
        "method": "Source-gated contiguous windows; prices become log returns; robust departures, persistence, volatility and coverage diagnostics.",
        "limits": ["Descriptive attention policy, not a calibrated probability, significance test or trading signal.",
                   "No detected departure is not proof of noise, safety or absence of risk.",
                   "Source permissions and provenance are caller declarations, not independently verified.",
                   "Correlated movements are not independent confirmation; no causal inference is made.",
                   "Market calendars, corporate actions, seasonality and economic materiality require caller review.",
                   "Missing availability timestamps prevent an as-published historical claim."],
        "execution_authority": False, "request_sha256": _hash(request), "policy_sha256": _hash(applied),
    }
