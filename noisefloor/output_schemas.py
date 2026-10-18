"""Discovery contracts for existing result payloads, not runtime coercion.

Success shapes come from the computational modules. MCP also returns the
existing {"error": string} object in structuredContent on tool errors. Keep
that branch in discovery without changing a calculation or an error response.
Schema validation belongs in development tests; the runtime has no dependency
on a general JSON Schema validator.
"""
from copy import deepcopy

from .schemas import MARKET_SOURCE, NARRATIVE_SOURCE


def scalar(kind, description, **keywords):
    return {"type": kind, "description": description, **keywords}


def text(description, **keywords):
    return scalar("string", description, **keywords)


def number(description, **keywords):
    return scalar("number", description, **keywords)


def count(description):
    return scalar("integer", description, minimum=0)


def flag(description, value=None):
    return scalar("boolean", description, **({} if value is None else {"const": value}))


def null(description):
    return scalar("null", description)


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}],
            "description": schema.get("description", "") + " Null means unavailable or not applicable, never zero."}


def array(items, description, **keywords):
    return {"type": "array", "items": items, "description": description, **keywords}


def record(properties, description, optional=(), **keywords):
    return {"type": "object", "description": description, "properties": properties,
            "required": [key for key in properties if key not in optional],
            "additionalProperties": False, **keywords}


STRINGS = array(text("One retained statement or identifier."), "Ordered statements or identifiers.")
NUMBERS = array(number("One finite numeric value."), "Numeric values in the documented order.")
MATRIX = array(NUMBERS, "Rows of a matrix in the declared series order.")
INTERVAL = array(number("Bound in the original input units."), "Lower and upper bounds, in that order.", minItems=2, maxItems=2)
HASH = text("SHA-256 of the canonicalized submitted request or applied policy.", pattern="^[0-9a-f]{64}$")
NO_GUARANTEE = null("No statistical or market-performance guarantee is made.")
DESCRIPTIVE = text("Descriptive assessment, not a probability or statistical certificate.", const="descriptive")
ERROR_SCHEMA = record({"error": text("Validation or computation failure message; inspect MCP isError or the HTTP status.")},
                      "Existing tool-error payload; no successful result fields are implied.")

ARM = record({
    "successes": count("Cumulative successful binary outcomes."),
    "n": count("Cumulative trials, including failures."),
    "rate": nullable(number("Observed success fraction; null when n is zero.")),
    "interval": array(number("Success-probability bound as a fraction."),
                      "Lower and upper confidence-sequence bounds.", minItems=2, maxItems=2),
}, "One experiment arm, stored under its caller-supplied label.")
AB_TEST = record({
    "decided": flag("Whether the two confidence sequences are disjoint."),
    "winner": nullable(text("Winning caller-supplied arm label; null when undecided.")),
    "reading": text("Interpretation of the current comparison."),
    "alpha": number("Total error budget split across the arms, as a fraction."),
    "safe_to_peek": flag("Repeated looks are allowed only under the stated Bernoulli assumptions.", True),
    "validity": text("Guarantees are conditional on the Bernoulli sampling model.", const="conditional_on_bernoulli_model"),
    "guarantee": text("Conditional confidence-sequence guarantee; not a market-performance claim."),
    "assumptions": STRINGS,
    "method": text("Confidence-sequence method and reference."),
    "observed_difference_pp": number("B's observed success rate minus A's, in percentage points; present only for undecided nonempty arms."),
    "what_would_decide_it": text("Qualitative guidance for an undecided comparison with data in both arms."),
}, "Two dynamically named arm objects plus comparison metadata. A and B are defaults, not fixed output keys.",
    optional=("observed_difference_pp", "what_would_decide_it"), additionalProperties=ARM)

EFFECT = record({
    "direction": text("Direction relative to the preceding reference median.", enum=["up", "down", "flat"]),
    "delta": nullable(number("Latest reading minus reference median, in input units.")),
    "robust_z": nullable(number("Departure divided by 1.4826 times reference MAD; unavailable at zero spread.")),
    "percentile": nullable(number("Percentage of reference readings strictly below the current value.")),
}, "Latest reading compared with the current post-reset reference.")
CHANGE = record({
    "n": count("Number of supplied observations."),
    "state": text("Current descriptive monitor state.", enum=["warming_up", "steady", "watch", "changed"]),
    "evidence": number("Compatibility alias of monitoring_statistic; not an e-value."),
    "monitoring_statistic": number("Descriptive Shiryaev-Roberts statistic."),
    "statistic_type": text("Monitor family.", const="shiryaev_roberts"),
    "is_e_value": flag("The statistic is not a calibrated e-value.", False),
    "validity": DESCRIPTIVE,
    "weighted": flag("Whether half-life weighting was requested."),
    "assumptions": STRINGS,
    "direction": nullable(text("Direction driving the positive monitor statistic.", enum=["up", "down"])),
    "effect": EFFECT,
    "effect_reference_size": count("Preceding observations in the current reference."),
    "change_points": array(count("Zero-based observation index."), "Historical alarm indices."),
    "watch_at": number("Descriptive watch threshold."),
    "changed_at": number("Descriptive alarm threshold."),
    "readings_since_change": nullable(count("Observations after the most recent alarm.")),
    "reading": text("Interpretation, including remaining warmup requirements."),
    "guarantee": NO_GUARANTEE,
    "interpretation": text("Limits on interpreting a threshold crossing."),
    "method": text("Monitor construction and one-sided/two-sided mode."),
}, "MCP change result; full per-reading states and evidence_series are deliberately omitted by this tool.")

COVERAGE = record({
    "n_pairs": count("Number of aligned metric/measurement-denominator pairs."),
    "validity": DESCRIPTIVE, "guarantee": NO_GUARANTEE,
    "is_causal": flag("Association does not establish causality.", False),
    "method": text("Descriptive denominator-conditioning method."), "caveats": STRINGS,
    "verdict": text("Legacy descriptive result code.", enum=["NOT_ENOUGH_HISTORY", "NO_MOVE", "UNCLEAR", "SAMPLING_ARTIFACT", "REAL"]),
    "assessment": text("Preferred interpretation of the legacy verdict.", enum=["insufficient_history", "no_supported_departure", "departure_from_zero_spread", "denominator_explanation_unidentified", "weak_linear_association", "consistent_with_denominator_association", "departure_survives_linear_conditioning"]),
    "reading": text("Explanation and limits of the result."),
    "metric_now": number("Latest metric in input units."),
    "sample_size_now": number("Latest declared measurement denominator."),
    "metric_departure": nullable(number("Robust standardized latest metric departure.")),
    "metric_delta": number("Latest metric minus historical median, in input units."),
    "zero_reference_spread": flag("Historical metric MAD is zero."),
    "sample_size_departure": nullable(number("Robust standardized denominator departure.")),
    "sample_size_change_pct": number("Denominator change as percent of its historical median."),
    "fit": record({"slope": number("Metric units per denominator unit."), "intercept": number("Metric units."),
                   "correlation": number("Historical linear correlation, rounded to three decimals.")}, "Historical linear association, when estimable."),
    "denominator_extrapolation": flag("Latest denominator falls outside the historical range."),
    "departure_after_conditioning": nullable(number("Robust standardized latest residual after linear conditioning.")),
}, "Short histories omit departure/fit fields; later branches retain only quantities that were computed.",
    optional=("metric_now", "sample_size_now", "metric_departure", "metric_delta", "zero_reference_spread",
              "sample_size_departure", "sample_size_change_pct", "fit", "denominator_extrapolation", "departure_after_conditioning"))

FORECAST_CONTRACT = {
    "ok": flag("False denotes insufficient history; forecast or scoring fields are then absent."),
    "validity": text("Coverage is empirical only.", const="empirical_only"),
    "guarantee": NO_GUARANTEE, "method": text("Shared forecast issuance and scoring method."),
    "caveats": STRINGS, "reading": text("Result interpretation or minimum-history requirement."),
}
PREDICTION = {
    "forecast": number("One-step random-walk point forecast in input units."),
    "interval": INTERVAL,
    "nominal_coverage": number("Requested target coverage as a fraction, not a guarantee."),
    "adaptive_alpha": number("Clipped adaptive miscoverage parameter as a fraction."),
    "error_history_size": count("Historical absolute errors used before issuing this forecast."),
}
FORECAST = record({**FORECAST_CONTRACT, **PREDICTION}, "One-step empirical forecast or insufficient-history result.", optional=tuple(PREDICTION),
                  allOf=[{"if": {"properties": {"ok": {"const": True}}}, "then": {"required": list(PREDICTION)}}])
FORECAST_RECORD = record({
    "index": count("Zero-based index of the observation scored after forecast issuance."), **PREDICTION,
    "actual": number("Observed value in input units."), "covered": flag("Actual value lies inside the issued interval."),
    "missed_by": number("Distance outside the interval in input units, zero when covered."),
    "score": number("Adaptive weighted interval score in input units; lower is better."),
    "baseline_score": number("Historical-quantile baseline weighted interval score in input units."),
}, "One prequential issue/observe/score record; future observations do not alter it.")
SCORE_FIELDS = {
    "n_forecasts": count("Number of issued forecasts scored."),
    "nominal_coverage": number("Target coverage fraction."), "empirical_coverage": number("Observed covered fraction."),
    "calibrated": flag("Compatibility alias of within_coverage_tolerance; descriptive only."),
    "within_coverage_tolerance": flag("Observed coverage is within 0.05 of the target."),
    "calibration_criterion": text("Descriptive tolerance rule, not a significance test."),
    "score": number("Mean adaptive weighted interval score."), "baseline_score": number("Mean baseline weighted interval score."),
    "beats_baseline": flag("Adaptive mean score is strictly below baseline."),
    "skill_vs_baseline": nullable(number("One minus adaptive/baseline score; null when baseline score is zero.")),
    "n_misses": count("Number of uncovered observations."),
    "records": array(FORECAST_RECORD, "Every scored observation."),
    "misses": array(FORECAST_RECORD, "Every uncovered observation."),
    "worst_misses": array(FORECAST_RECORD, "At most five misses sorted by distance outside the interval.", maxItems=5),
    "records_complete": flag("All scored observations are included.", True),
}
SCORES = record({**FORECAST_CONTRACT, **SCORE_FIELDS}, "Forecast replay scores or insufficient-history result.", optional=tuple(SCORE_FIELDS),
                allOf=[{"if": {"properties": {"ok": {"const": True}}}, "then": {"required": list(SCORE_FIELDS)}}])
MULTIPLE = record({
    "selected": array(text("Caller-supplied metric name."), "Names selected only when the caller attests valid e-values."),
    "alpha": number("Requested expected false-discovery proportion bound, conditional on valid e-values."),
    "k": count("Number selected."), "n_watched": count("Number of supplied metrics."),
    "threshold": nullable(number("e-BH threshold; null without the caller's validity attestation.")),
    "board_evidence": nullable(number("Mean supplied valid e-values; null without attestation.")),
    "descriptive_mean": number("Mean supplied values, without certifying their statistical construction."),
    "ranked": array(record({"metric": text("Caller-supplied name."), "evidence": number("Supplied nonnegative value.")}, "One ranked input."), "Inputs sorted by decreasing value then name."),
    "valid_evalues": flag("Caller attestation, not a server certification."),
    "validity": text("Selection contract.", enum=["conditional_on_valid_evalues", "unverified_inputs"]),
    "guarantee": nullable(text("Conditional e-BH statement; null for unverified inputs.")),
    "assumptions": STRINGS, "reading": text("Selection interpretation."),
    "method": text("Conditional e-BH or descriptive ranking."), "note": text("Selection requests review, not trading."),
}, "A complete nonempty supplied-family result, including unverified rankings. Empty input is a tool error.")

MARKET_POLICY = record({"baseline_points": count("Earlier transformed readings."), "recent_points": count("Latest transformed readings."),
                        "move_threshold": number("Robust-spread units."), "volatility_ratio": number("Recent/baseline spread ratio."),
                        "coverage_drop_fraction": number("Absolute measurement-coverage change fraction.")}, "Fully applied policy, including defaults.")
SPECTRAL_POLICY = record({"window_points": count("Aligned intervals per window."), "step_points": count("Intervals between window ends."),
                          "max_windows": count("Maximum returned windows.")}, "Fully applied rolling policy.")
SOURCE_RECORD = {
    "id": text("Caller-supplied series identifier."), "kind": text("Caller-declared measurement kind."),
    "unit": text("Caller-declared input unit."), "source": MARKET_SOURCE, "group": text("Declared group, defaulting to series id."),
    "reasons": STRINGS, "warnings": STRINGS,
    "observed_at": nullable(text("Last supplied observation timestamp; null for no observations.")),
    "available_at": nullable(text("Last supplied availability timestamp, if present.")),
    "age_seconds": nullable(number("Cutoff minus last observation timestamp in seconds; may be negative for future data.")),
    "gap_indices": array(count("Zero-based observation index."), "Declared-calendar gap positions."),
    "missing_points": count("Null-valued observations."),
    "availability_timestamps_complete": flag("Every supplied observation declares availability and the series is nonempty."),
    "as_published_history_verified": flag("Caller timestamps do not verify historical availability.", False),
}
MARKET_DIAGNOSTICS = record({
    **{key: number(description) for key, description in {
        "baseline_median": "Baseline center in analyzed units.", "recent_median": "Recent center in analyzed units.",
        "median_change": "Recent minus baseline median in analyzed units.", "latest_change": "Latest minus baseline median in analyzed units.",
        "baseline_robust_spread": "Baseline MAD times 1.4826 in analyzed units.", "recent_robust_spread": "Recent MAD times 1.4826 in analyzed units.",
        "representation_tolerance": "Log-price arithmetic tolerance in analyzed units, zero for direct measurements.",
    }.items()},
    "robust_departure": nullable(number("Median change in baseline robust-spread units.")),
    "latest_robust_departure": nullable(number("Latest change in baseline robust-spread units.")),
    "volatility_ratio": nullable(number("Recent/baseline robust spread; null at zero baseline spread.")),
    "persistence_points": count("Recent readings with a supported same-direction departure."),
    "representation_tolerance_basis": text("Arithmetic source of the tolerance."),
    "lag_one_correlation": nullable(number("Lag-one correlation, when estimable.")),
    "coverage": record({"status": text("Measurement-denominator availability or change.", enum=["not_supplied", "incomplete", "changed", "stable"]),
                         "relative_change": nullable(number("Recent minus baseline median denominator divided by baseline."))}, "Measurement coverage, not trading volume."),
    "monitor": record({"state": text("Descriptive monitor state."), "statistic": number("Raw monitoring statistic, not an e-value."),
                        "statistic_type": text("Monitor family.", const="shiryaev_roberts"), "is_e_value": flag("Not a calibrated e-value.", False),
                        "change_points": array(count("Zero-based window position."), "Alarm positions in the analyzed window."),
                        "guarantee": NO_GUARANTEE}, "Descriptive monitor of the analyzed window."),
    "window_start": text("First analyzed observation timestamp."), "window_end": text("Last analyzed observation timestamp."),
}, "Computed only for source-eligible series with sufficient contiguous history.")
MARKET_SERIES = record({
    **SOURCE_RECORD, "status": text("Per-series assessment state.", enum=["insufficient_visibility", "review", "watch", "no_supported_departure"]),
    "observation_count": count("Supplied observation count."), "contiguous_points": count("Usable transformed tail points."),
    "required_points": count("Baseline plus recent point requirement."),
    "transform": text("price becomes log_return; all other kinds use identity.", enum=["log_return", "identity"]),
    "analyzed_unit": text("log_return for prices, otherwise the declared unit."),
    "diagnostics": nullable(MARKET_DIAGNOSTICS), "priority": nullable(count("Review=2, watch=1, no supported departure=0; null when ineligible.")),
    "is_e_value": flag("No calibrated e-value.", False), "validity": DESCRIPTIVE, "guarantee": NO_GUARANTEE,
    "analyzed_points": count("Points actually used; omitted for insufficient visibility."),
    "reading": text("Interpretation or visibility-gap explanation."),
}, "Every supplied series is retained, including rejected sources and missing histories.", optional=("analyzed_points",))
MARKET = record({
    "schema": text("Result contract identifier.", const="noisefloor.market-assessment.v1"), "as_of": text("Caller-supplied UTC cutoff."),
    "status": text("Aggregate visibility state.", enum=["insufficient_visibility", "partial", "assessed"]),
    "policy": MARKET_POLICY, "series": array(MARKET_SERIES, "All supplied series in input order."),
    "attention_queue": array(record({"id": text("Series id."), "status": text("Review priority state.", enum=["review", "watch"]),
                                      "reasons": STRINGS, "group": text("Declared group.")}, "One prioritized series."), "Review/watch series sorted by priority then id."),
    "visibility_gaps": STRINGS,
    "related_movements": array(record({"series": array(text("Series id."), "Pair of series ids.", minItems=2, maxItems=2),
                                         "correlation": number("Aligned Pearson correlation."), "aligned_points": count("Shared timestamps."),
                                         "independent_confirmation": flag("Correlation is not independent corroboration.", False)}, "Strong aligned relationship."), "Relationships with absolute correlation at least 0.85."),
    "coverage": record({"requested_series": count("Supplied series."), "assessed_series": count("Series passing visibility gates."),
                         "complete_for_supplied_series": flag("Nonempty supplied set with no visibility gaps."),
                         "market_universe_complete": flag("Market collection completeness is unknown.", False)}, "Coverage of only the supplied set."),
    "method": text("Assessment construction."), "limits": STRINGS,
    "execution_authority": flag("No trading authority.", False), "request_sha256": HASH, "policy_sha256": HASH,
}, "Market attention and explicit source/coverage gaps; no absent evidence is replaced by zero.")

NARRATIVE_CLUSTER = record({
    "cluster_id": text("Deterministic cluster id."), "representative_id": text("Selected representative event id."),
    "title": text("Untrusted caller headline text."), "entities": STRINGS, "published_at": text("Representative publication timestamp."),
    "source": NARRATIVE_SOURCE, "member_ids": STRINGS, "report_count": count("Member events."),
    "declared_source_families": STRINGS, "independence_verified": flag("Source independence is not verified.", False),
    "truth_assessed": flag("No truth assessment.", False), "impact_assessed": flag("No impact assessment.", False),
    "relevance": text("Caller-entity focus classification.", enum=["unmapped", "matches_focus", "not_filtered"]), "event_keys": STRINGS,
}, "One wording cluster, not independent corroboration or truth.")
NARRATIVE_EVENT = record({
    "id": text("Caller event id."), "status": text("Retained event disposition.", enum=["not_yet_available", "stale", "outside_focus", "unmapped_entity", "representative", "repeated_wording"]),
    "cluster_id": nullable(text("Assigned cluster id; null for excluded events.")),
    "published_at": text("Supplied publication timestamp."), "available_at": nullable(text("Supplied availability timestamp, if present.")),
    "source": NARRATIVE_SOURCE, "entities": STRINGS, "title": text("Untrusted caller headline."),
}, "Every supplied event remains visible with a disposition.")
NARRATIVE = record({
    "schema": text("Result contract identifier.", const="noisefloor.narrative-triage.v1"), "as_of": text("Caller-supplied UTC cutoff."), "focus": STRINGS,
    "status": text("Whether any supplied events were eligible.", enum=["triaged", "insufficient_visibility"]),
    "policy": record({"max_age_seconds": number("Maximum event age in seconds."), "max_items": count("Maximum queue length."),
                       "similarity_threshold": number("Token-set similarity threshold.")}, "Fully applied policy."),
    "attention_queue": array(NARRATIVE_CLUSTER, "Prioritized bounded queue."), "clusters": array(NARRATIVE_CLUSTER, "All eligible wording clusters."),
    "events": array(NARRATIVE_EVENT, "All events sorted by id."),
    "conflict_candidates": array(record({"event_key": text("Shared caller-defined event identity."), "cluster_ids": STRINGS,
                                         "reason": text("Review reason.", const="different_wording_or_quantities_requires_review")}, "Distinct wording sharing an event key."), "Potential conflicts requiring review."),
    "deferred_cluster_ids": STRINGS,
    "counts": record({key: count(description) for key, description in {"supplied": "Input events.", "eligible_for_triage": "Eligible events.",
                      "distinct_wording_clusters": "Clusters.", "repeated_wording": "Eligible events minus clusters.",
                      "queued": "Queued clusters.", "deferred": "Clusters beyond queue capacity."}.items()}, "Complete event/cluster counts."),
    "limits": STRINGS, "source_universe_complete": flag("Collection completeness is unknown.", False),
    "execution_authority": flag("No trading authority.", False), "request_sha256": HASH, "policy_sha256": HASH,
}, "Headline repetition and review-priority result, never a truth or market-impact score.")

SPECTRAL_SERIES = record({
    **SOURCE_RECORD, "transform": text("Transformation applied to the contiguous tail.", enum=["log_return", "identity_return", "first_difference"]),
    "measurement_coverage": record({"status": text("Declared denominator state.", enum=["not_supplied", "incomplete", "variable", "constant_declared_count"]),
                                     "declared_points": count("Non-null declared denominators."), "tail_points": count("Contiguous tail observations."),
                                     "minimum": nullable(number("Minimum declared denominator.")), "maximum": nullable(number("Maximum declared denominator.")),
                                     "sampling_adjustment_applied": flag("No adjustment is made.", False)}, "Caller-declared measurement coverage."),
    "contiguous_intervals": count("Transformed eligible intervals; zero when source gates fail."),
}, "Panel member retained even when its source or history is ineligible.")
SNAPSHOT_BASE = {"window_start": text("First interval start in UTC."), "window_end": text("Last interval end in UTC."),
                 "observations": count("Aligned intervals in this window."), "series_ids": STRINGS,
                 "status": text("Window assessment state.", enum=["assessed", "insufficient_visibility"]), "reasons": STRINGS}
SNAPSHOT_FIELDS = {
    "correlation": MATRIX, "eigenvalues": NUMBERS,
    "leading_variance_share": number("Largest eigenvalue divided by trace."),
    "participation_ratio": number("Squared trace divided by sum of squared eigenvalues."),
    "effective_rank": number("Exponential entropy of normalized positive eigenvalues."),
    "numerical_rank": count("Eigenvalues above 1e-10."), "leading_eigengap": number("First minus second eigenvalue."),
    "leading_loadings": {"type": "object", "description": "Leading eigenvector components keyed by caller series ids; sign is normalized, not an economic factor label.", "additionalProperties": number("Eigenvector component.")},
    "lag_one_correlations": array(record({"id": text("Series id."), "correlation": nullable(number("Lag-one correlation, when estimable."))}, "One series lag diagnostic."), "Lag diagnostics in series order."),
    "reference": record({"name": text("Reference model.", const="marchenko_pastur_iid_unit_variance"), "q": number("Series count divided by sample degrees of freedom."),
                          "sample_degrees_of_freedom": count("Window observations minus one."), "lower_edge": number("Asymptotic lower reference edge."), "upper_edge": number("Asymptotic upper reference edge."),
                          "above_upper_count": count("Eigenvalues above upper edge."), "below_lower_count": count("Eigenvalues below lower edge."), "in_band_count": count("Eigenvalues inside the reference band."),
                          "finite_sample_calibrated": flag("Not a finite-sample significance test.", False), "p_value": null("No calibrated p-value."), "is_e_value": flag("Not an e-value.", False)}, "Idealized reference, not evidence of alpha or certified factors."),
    "shrinkage_candidate": record({"status": text("Candidate maturity.", const="experimental"), "correlation": MATRIX,
                                    "method": text("Construction.", const="equalize_in_band_eigenvalues_then_restore_unit_diagonal"),
                                    "low_and_high_outliers_preserved_before_diagonal_normalization": flag("Outliers retained before diagonal normalization.", True),
                                    "positive_definite_guaranteed": flag("Singular candidates remain possible.", False), "out_of_sample_validated": flag("No held-out validation.", False)}, "Experimental correlation candidate, not a validated risk model."),
}
SNAPSHOT = record({**SNAPSHOT_BASE, **SNAPSHOT_FIELDS}, "Window with all metrics absent when a series is constant or numerically unresolved.", optional=tuple(SNAPSHOT_FIELDS),
                  allOf=[{"if": {"properties": {"status": {"const": "assessed"}}}, "then": {"required": list(SNAPSHOT_FIELDS)}}])
SPECTRAL = record({
    "schema": text("Result contract identifier.", const="noisefloor.spectral-assessment.v1"), "as_of": text("Caller cutoff."),
    "status": text("Window/visibility result.", enum=["insufficient_visibility", "assessed", "partial"]),
    "maturity": text("Research maturity.", const="research"), "validity": DESCRIPTIVE, "guarantee": NO_GUARANTEE,
    "is_e_value": flag("No calibrated e-value.", False), "policy": SPECTRAL_POLICY,
    "series": array(SPECTRAL_SERIES, "All requested series sorted by id."), "reasons": STRINGS,
    "windows": array(SNAPSHOT, "Complete recent windows in chronological order."), "latest": nullable(SNAPSHOT),
    "dynamics": array(record({"from": text("Previous window-end timestamp."), "to": text("Current window-end timestamp."),
                               "leading_variance_share_change": number("Current minus previous leading variance share."),
                               "eigenvalue_rms_change": number("RMS eigenvalue difference."), "correlation_rms_change": number("RMS matrix-entry difference."),
                               "leading_vector_squared_overlap": nullable(number("Squared overlap; null when the leading mode is degenerate.")),
                               "leading_vector_comparison": text("Whether vector overlap is resolved.", enum=["descriptive", "unresolved_degenerate_mode"]),
                               "calibrated_regime_change": flag("No calibrated change decision.", False)}, "Adjacent assessed-window comparison."), "Only pairs of assessed windows are compared."),
    "alignment": record({"method": text("Alignment method.", const="common_contiguous_interval_suffix"), "aligned_points": count("Aligned contiguous intervals."),
                           "first_observation_sets_interval_start": flag("First observation sets the first interval start.", True), "filled_points": count("Always zero; data are never filled."),
                           "unmatched_intervals": count("Union intervals absent from at least one member."),
                           "discarded_common_intervals": count("Common intervals before the last mismatch.")}, "Alignment diagnostics; mismatch counts are absent when early gates stop alignment.", optional=("unmatched_intervals", "discarded_common_intervals")),
    "coverage": record({"requested_series": count("Requested panel members."), "complete_for_supplied_series": flag("All returned windows assessed for the entire supplied panel."),
                         "market_universe_complete": flag("No claim of market-wide completeness.", False)}, "Supplied-panel coverage only."),
    "limits": STRINGS, "execution_authority": flag("No trading authority.", False), "request_sha256": HASH, "policy_sha256": HASH,
}, "Source-gated spectral research, with explicit unresolved and partial windows.")
DYSON = record({
    "schema": text("Result contract identifier.", const="noisefloor.dyson-reference.v1"), "status": text("Synthetic output.", const="synthetic"),
    "maturity": text("Research maturity.", const="research"), "validity": text("Synthetic reference only.", const="synthetic_reference"), "guarantee": NO_GUARANTEE,
    "parameters": record({"dimension": count("Matrix dimension N."), "steps": count("Number of increments."), "dt": number("Synthetic time step, not seconds."),
                           "seed": count("Replay seed for this Python runtime.")}, "Fully applied parameters."),
    "beta": scalar("integer", "Real-symmetric reference beta.", const=1),
    "path": array(record({"time": number("Synthetic elapsed time."), "eigenvalues": NUMBERS}, "Ordered spectrum at one synthetic time."), "Initial zero spectrum followed by every simulated step."),
    "initial_condition": text("Initial matrix.", const="zero_matrix"), "mean_reversion": flag("Unconfined Brownian reference.", False),
    "normalization": text("Increment variance convention."), "market_calibrated": flag("Not calibrated to markets.", False),
    "is_covariance_model": flag("Negative eigenvalues are possible; not a covariance model.", False),
    "execution_authority": flag("No trading authority.", False), "request_sha256": HASH, "limits": STRINGS,
}, "Seeded synthetic eigenvalue path, never market observations or a covariance forecast.")

SUCCESS_SCHEMAS = {
    "ab_test": AB_TEST, "did_it_change": CHANGE, "real_or_sampling": COVERAGE,
    "forecast_next": FORECAST, "score_forecasts": SCORES, "which_metrics_matter": MULTIPLE,
    "market_assessment": MARKET, "narrative_triage": NARRATIVE,
    "spectral_assessment": SPECTRAL, "dyson_reference": DYSON,
}


def success_schema(name):
    """Return an owned schema for the unwrapped successful result (REST 200)."""
    return deepcopy(SUCCESS_SCHEMAS[name])


def output_schema(name):
    """MCP structuredContent can be success or the existing tool-error object.

    Keep named properties at the root for discovery clients. Branch requirements
    remain inside anyOf because an error has none of the success fields. An A/B
    arm may legally be called 'error', so that success property is an arm object,
    whereas the tool-error branch contains a string. Never reserve a new label.
    """
    success = success_schema(name)
    properties = deepcopy(success["properties"])
    properties["error"] = ({"anyOf": [deepcopy(ERROR_SCHEMA["properties"]["error"]), deepcopy(ARM)],
                            "description": "A string on tool error, or a successful arm object when the caller labels that arm error; inspect MCP isError."}
                           if name == "ab_test" else deepcopy(ERROR_SCHEMA["properties"]["error"]))
    # Property types are declared once at the root. The success branch retains
    # presence/closure/conditional rules without doubling every nested schema
    # in tools/list. Empty property schemas here intentionally defer to the root.
    branch = {key: value for key, value in success.items() if key not in ("description", "properties")}
    branch["properties"] = {key: {} for key in success["properties"]}
    return {"type": "object", "description": success["description"] + " Tool errors retain the separate error-message shape.",
            "properties": properties, "anyOf": [branch, deepcopy(ERROR_SCHEMA)]}
