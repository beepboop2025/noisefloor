"""Correlation spectra and rolling concentration diagnostics, not trade signals.

Random-matrix reference edges describe an ideal independent, standardized null.
They are neither finite-sample significance thresholds nor evidence of alpha.
"""
from __future__ import annotations

import math
from copy import deepcopy

from ._linalg import eigh, gram, normalized
from .market import _hash, _number, _prepare_series
from .schemas import SPECTRAL_REQUEST, parse_utc, validate

DEFAULT_POLICY = {"window_points": 60, "step_points": 20, "max_windows": 4}
LIMITS = [
    "Research/descriptive diagnostics; no p-value, e-value, causal claim or trading authority.",
    "Marchenko-Pastur edges are an asymptotic iid unit-variance reference, not a calibrated significance test.",
    "Serial dependence, heavy tails, volatility changes and selected windows can invalidate the reference.",
    "An outlying mode is a candidate for review, not a proven risk factor; in-band does not prove noise.",
    "The shrinkage candidate is experimental and needs held-out comparison with raw and shrinkage baselines.",
    "Correlation standardizes units; it does not establish economic or jurisdictional comparability.",
    "Source rights, availability and interval alignment are caller declarations, not attestations.",
    "Measurement coverage is reported, not adjusted away; sampling changes can produce shared movements.",
    "Rolling snapshots use the supplied vintage; they are not verified historical decision-time releases.",
    "Ordered eigenvalue paths are descriptive; financial correlations are not assumed to follow Dyson dynamics.",
]


def _clock(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


def _prepare(item, now):
    ready = _prepare_series(item, now)
    tail, obs = ready["tail"], ready["observations"]
    transform = "log_return" if item["kind"] == "price" else "identity_return" if item["kind"] == "return" else "first_difference"
    record = {"id": item["id"], "kind": item["kind"], "unit": item["unit"],
              "source": deepcopy(item["source"]), "group": item.get("group", item["id"]),
              "transform": transform, "reasons": ready["issues"], "warnings": ready["warnings"],
              "observed_at": obs[-1]["observed_at"] if obs else None,
              "available_at": obs[-1].get("available_at") if obs else None,
              "age_seconds": ready["age"], "gap_indices": ready["gaps"],
              "missing_points": sum(p["value"] is None for p in obs),
              "availability_timestamps_complete": bool(obs) and all("available_at" in p for p in obs),
              "as_published_history_verified": False}
    supplied = [point.get("sample_size") for point in tail]
    counts = [count for count in supplied if count is not None]
    record["measurement_coverage"] = {
        "status": "not_supplied" if not any("sample_size" in point for point in tail)
        else "incomplete" if len(counts) != len(tail)
        else "variable" if len(set(counts)) > 1 else "constant_declared_count",
        "declared_points": len(counts), "tail_points": len(tail),
        "minimum": min(counts) if counts else None, "maximum": max(counts) if counts else None,
        "sampling_adjustment_applied": False,
    }
    if record["measurement_coverage"]["status"] in {"variable", "incomplete"}:
        record["warnings"].append("review_measurement_coverage")
    values, precision = {}, {}
    if not ready["issues"]:
        # Match BOTH endpoints, not just the close. A two-day move cannot be
        # correlated as if it were the same observation as a one-day move.
        for before, after in zip(tail, tail[1:]):
            interval = (parse_utc(before["observed_at"]), parse_utc(after["observed_at"]))
            if transform == "log_return":
                a, b = math.log(before["value"]), math.log(after["value"])
                value = b - a
                precision[interval] = 4 * (math.ulp(a) + math.ulp(b) + math.ulp(value))
            elif transform == "identity_return":
                value = float(after["value"])
            else:
                value = after["value"] - before["value"]
            values[interval] = _number(value)
    record["contiguous_intervals"] = len(values)
    return record, values, precision


def _snapshot(columns, precisions, ids, intervals):
    n, t = len(ids), len(intervals)
    result = {"window_start": _clock(intervals[0][0]), "window_end": _clock(intervals[-1][1]),
              "observations": t, "series_ids": ids, "status": "assessed", "reasons": []}
    unit_columns = []
    for name, values, tolerance in zip(ids, columns, precisions):
        column = normalized(values)
        # A constant geometric growth path has representation noise in log()
        # differences. Do not amplify that into an arbitrary correlation.
        if column is None or (tolerance and max(values) - min(values) <= 2 * tolerance):
            result["reasons"].append("constant_or_unresolved_series:" + name)
        unit_columns.append(column)
    if result["reasons"]:
        result["status"] = "insufficient_visibility"
        return result
    correlation = gram(unit_columns)
    values, vectors = eigh(correlation)
    if min(values) < -1e-10:
        raise ValueError("correlation matrix failed positive-semidefinite verification")
    values = [max(0.0, x) for x in values]
    q = n / (t - 1)  # explicit degrees-of-freedom convention after demeaning
    lower, upper = (1 - math.sqrt(q)) ** 2, (1 + math.sqrt(q)) ** 2
    in_band = [k for k, value in enumerate(values) if lower <= value <= upper]
    candidate = list(values)
    if in_band:
        mean = math.fsum(values[k] for k in in_band) / len(in_band)
        for k in in_band:
            candidate[k] = mean
    reconstructed = [[math.fsum(vectors[i][k] * candidate[k] * vectors[j][k] for k in range(n))
                      for j in range(n)] for i in range(n)]
    diagonal = [math.sqrt(reconstructed[i][i]) for i in range(n)]
    cleaned = [[reconstructed[i][j] / (diagonal[i] * diagonal[j]) for j in range(n)] for i in range(n)]
    leading = [row[0] for row in vectors]
    sign = 1 if leading[max(range(n), key=lambda i: abs(leading[i]))] >= 0 else -1
    leading = [sign * x for x in leading]
    trace = math.fsum(values)
    probabilities = [value / trace for value in values if value > 0]
    lag_one = []
    for name, column in zip(ids, columns):
        before, after = normalized(column[:-1]), normalized(column[1:])
        lag_one.append({"id": name, "correlation":
                        math.fsum(a * b for a, b in zip(before, after)) if before and after else None})
    result.update({
        "correlation": correlation, "eigenvalues": values,
        "leading_variance_share": values[0] / trace,
        "participation_ratio": trace * trace / math.fsum(x * x for x in values),
        "effective_rank": math.exp(-math.fsum(p * math.log(p) for p in probabilities)),
        "numerical_rank": sum(x > 1e-10 for x in values),
        "leading_eigengap": values[0] - values[1],
        "leading_loadings": dict(zip(ids, leading)),
        "lag_one_correlations": lag_one,
        "reference": {"name": "marchenko_pastur_iid_unit_variance", "q": q,
                      "sample_degrees_of_freedom": t - 1, "lower_edge": lower, "upper_edge": upper,
                      "above_upper_count": sum(x > upper for x in values),
                      "below_lower_count": sum(x < lower for x in values),
                      "in_band_count": len(in_band), "finite_sample_calibrated": False,
                      "p_value": None, "is_e_value": False},
        "shrinkage_candidate": {"status": "experimental", "correlation": cleaned,
                                "method": "equalize_in_band_eigenvalues_then_restore_unit_diagonal",
                                "low_and_high_outliers_preserved_before_diagonal_normalization": True,
                                "positive_definite_guaranteed": False,
                                "out_of_sample_validated": False},
    })
    return result


def _dynamics(previous, current):
    n = len(current["series_ids"])
    overlap = None
    # Sign ambiguity cancels in squared overlap. Degenerate leading modes
    # have no unique vector, so never report their rotation as a regime shift.
    if min(previous["leading_eigengap"], current["leading_eigengap"]) > 1e-6:
        overlap = min(1.0, math.fsum(previous["leading_loadings"][name] * current["leading_loadings"][name]
                                     for name in current["series_ids"]) ** 2)
    return {"from": previous["window_end"], "to": current["window_end"],
            "leading_variance_share_change": current["leading_variance_share"] - previous["leading_variance_share"],
            "eigenvalue_rms_change": math.sqrt(math.fsum((a-b) ** 2 for a, b in
                                                        zip(previous["eigenvalues"], current["eigenvalues"])) / n),
            "correlation_rms_change": math.sqrt(math.fsum((previous["correlation"][i][j] - current["correlation"][i][j]) ** 2
                                                          for i in range(n) for j in range(n)) / (n*n)),
            "leading_vector_squared_overlap": overlap,
            "leading_vector_comparison": "descriptive" if overlap is not None else "unresolved_degenerate_mode",
            "calibrated_regime_change": False}


def assess(series: list[dict], *, as_of: str, policy: dict | None = None) -> dict:
    """Analyze one comparable panel; all supplied members must pass the gates.

    Prices become log returns, supplied returns stay returns, other levels become
    first differences. A common contiguous suffix aligns exact interval pairs.
    Missing or rejected members are never silently removed from the universe.
    """
    request = {"series": series, "as_of": as_of}
    if policy is not None:
        request["policy"] = policy
    validate(request, SPECTRAL_REQUEST)
    if len({s["id"] for s in series}) != len(series):
        raise ValueError("series ids must be unique")
    applied = {**DEFAULT_POLICY, **(policy or {})}
    ordered = sorted(series, key=lambda s: s["id"])
    prepared = [_prepare(item, parse_utc(as_of)) for item in ordered]
    records = [entry[0] for entry in prepared]
    reasons = []
    result = {"schema": "noisefloor.spectral-assessment.v1", "as_of": as_of,
              "status": "insufficient_visibility", "maturity": "research",
              "validity": "descriptive", "guarantee": None, "is_e_value": False,
              "policy": applied, "series": records, "reasons": reasons,
              "windows": [], "latest": None, "dynamics": [],
              "alignment": {"method": "common_contiguous_interval_suffix", "aligned_points": 0,
                            "first_observation_sets_interval_start": True, "filled_points": 0},
              "coverage": {"requested_series": len(series), "complete_for_supplied_series": False,
                           "market_universe_complete": False},
              "limits": list(LIMITS), "execution_authority": False,
              "request_sha256": _hash(request), "policy_sha256": _hash(applied)}
    if len(series) < 2:
        reasons.append("at_least_two_series_required")
    if applied["window_points"] <= len(series):
        reasons.append("window_must_exceed_series_count_for_reference")
    if any(entry["reasons"] for entry in records):
        reasons.append("one_or_more_series_ineligible")
    if any(not entry[1] for entry in prepared):
        reasons.append("insufficient_contiguous_intervals")
    if reasons:
        return result
    sets = [set(entry[1]) for entry in prepared]
    common, union = set.intersection(*sets), set.union(*sets)
    mismatched = union - common
    # A gap on ANY member cuts the whole panel; pairwise deletion would create
    # matrices whose entries describe different samples and can even be indefinite.
    if mismatched:
        last_bad_end = max(interval[1] for interval in mismatched)
        common = {interval for interval in common if interval[0] >= last_bad_end}
    intervals = sorted(common)
    result["alignment"].update(aligned_points=len(intervals), unmatched_intervals=len(mismatched),
                                discarded_common_intervals=len(set.intersection(*sets)) - len(intervals))
    if len(intervals) < applied["window_points"]:
        reasons.append("insufficient_aligned_contiguous_history")
        return result
    ids = [entry["id"] for entry in records]
    ends = sorted(len(intervals) - i * applied["step_points"] for i in range(applied["max_windows"])
                  if len(intervals) - i * applied["step_points"] >= applied["window_points"])
    for end in ends:
        selected = intervals[end-applied["window_points"]:end]
        columns = [[entry[1][interval] for interval in selected] for entry in prepared]
        precisions = [max((entry[2].get(interval, 0) for interval in selected), default=0) for entry in prepared]
        result["windows"].append(_snapshot(columns, precisions, ids, selected))
    for previous, current in zip(result["windows"], result["windows"][1:]):
        if previous["status"] == current["status"] == "assessed":
            result["dynamics"].append(_dynamics(previous, current))
    result["latest"] = result["windows"][-1]
    latest_ok = result["latest"]["status"] == "assessed"
    all_ok = all(window["status"] == "assessed" for window in result["windows"])
    result["status"] = "assessed" if all_ok else "partial" if latest_ok else "insufficient_visibility"
    if not all_ok:
        reasons.append("one_or_more_windows_unresolved")
    result["coverage"]["complete_for_supplied_series"] = all_ok
    return result
