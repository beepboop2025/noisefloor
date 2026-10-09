# Statistical methods and interpretation

Noisefloor separates a descriptive monitoring score, empirical forecast quality,
and model-dependent statistical evidence. A numerical score does not establish
that a movement is economically important or actionable. An absent alarm does
not prove that a market is stable or that a narrative is noise.

## Contracts

| Module | Output | What it supports | What it does not establish |
| --- | --- | --- | --- |
| `experiment` | Bernoulli confidence sequence | Repeated inspection of cumulative Bernoulli counts under the stated sampling model | Validity for changing win probabilities, selected windows or arbitrary trading outcomes |
| `change` | Shiryaev–Roberts monitoring statistic and descriptive threshold crossings | Review of departures against prior observations | A valid e-value, universal false-alarm rate or profitability |
| `coverage` | Historical linear denominator association | Identification of a possible measurement explanation | Causality, exclusion of nonlinear/sample-composition effects or proof a market move is real |
| `forecast` | Adaptive historical-error interval and complete prequential ledger | Measured coverage, interval width and proper-score comparison | Arbitrary-shift coverage or a guarantee for the next observation |
| `multiple` | Descriptive ranking by default; conditional e-BH when explicitly enabled | One-family FDR control if every supplied input is a valid e-value | Certification of inputs, automatic repeated-board control or a trade recommendation |
| `spectral` | Correlation spectrum, concentration and iid reference edges | Review of shared movements in a gated, aligned panel | Finite-sample significance, certified factors, alpha or superior out-of-sample covariance |
| `dyson` | Seeded real-symmetric Brownian eigenvalue path | A synthetic random-matrix reference | Positive-semidefinite covariance, calibrated market dynamics or a forecast |

The [spectral method and product contract](SPECTRAL_RISK.md) defines transforms,
interval alignment, shrinkage, matrix normalization and research promotion tests.

Each public result publishes `validity` and `guarantee`; the latter is `null`
where no theorem is established for the supplied data. Input numbers must be
finite, counts must be nonnegative integers, rates must have positive
measurement denominators, and paired arrays must align exactly. Missingness is
not replaced by zero or silently removed.

## Change monitoring is not e-value generation

The detector computes `R_t = (1 + R_(t-1)) * bet(rank_t)`, restarts after a
threshold crossing, and maintains separate directional scores. The compatibility
field `evidence` is an alias of `monitoring_statistic`, with
`statistic_type="shiryaev_roberts"` and `is_e_value=false`.

A sum over possible change starts is not normalized like an e-value. Supplying
this statistic to e-BH can exceed the claimed false-discovery rate even for
independent unchanged observations. Distribution-free ranks still need
exchangeability; dependence, selected windows, volatility changes and geometric
weighting must not inherit an unconditional guarantee. The historical doubled
two-sided threshold remains a descriptive sensitivity setting. Even under the
appropriate ideal rank assumptions, a union over two thresholds of 1,000 would
produce an `n/500` bound, not `n/1000`.

[Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415)
describes assumptions and coverage loss under departures from exchangeability;
its results are not a blanket guarantee for arbitrary weighted market series.

## Forecasts are issued and scored by the same engine

The order is: issue from past errors, observe the next value, record coverage
and score, then update the error pool and adaptive parameter. Calling
`next_value(history[:t])` produces exactly the forecast at index `t` in
`score(history)["records"]`. Later observations cannot alter earlier records.

The clipped feedback heuristic uses finite interpolated historical-error
quantiles. It deliberately makes no arbitrary-shift coverage claim. For example,
in the adverse sequence `value[t] = t*t`, the next absolute change always
exceeds the earlier changes and all issued intervals can miss. The ledger
publishes this failure instead of claiming calibration.

`calibrated` is retained as a compatibility alias for
`within_coverage_tolerance`: observed coverage within five percentage points of
the target. This is a descriptive tolerance, not a statistical certificate.
`records` and `misses` contain every scored observation and miss;
`worst_misses` is only a five-item convenience summary. All evaluate the supplied
series, not a verified historical market data feed. All intervals share one
observation horizon; callers must align timestamps, sessions and revisions.

[Adaptive conformal inference](https://arxiv.org/abs/2106.00170) and
[decaying-step online conformal prediction](https://arxiv.org/abs/2402.01139)
provide coverage-frequency results for their specified algorithms. The bounded
heuristic here must not be represented as that theorem. WIS remains a proper
scoring rule for the reported central intervals; good WIS is not trading skill.

## Conditional multiple testing

`multiple.select(values)` ranks values but returns no selections and no FDR
claim. To use e-BH, call `multiple.select(values, valid_evalues=True)` only after
establishing that every value is nonnegative with expectation at most one under
its defined null. Define the nulls, windows, method and any data selection.
Caller confirmation records an assumption; noisefloor cannot verify it from a
number. Raw detector statistics, returns and model confidence are not valid
substitutes.

[e-BH](https://arxiv.org/abs/2009.02824) allows arbitrary dependence **between
valid e-values**. This does not remove the requirements on each input or prove
error control over repeated, adaptively selected hypothesis families. Arithmetic
merging has the same conditional input requirements.

## Denominator diagnostics and financial use

A zero-spread baseline followed by a different value reports an unstandardized
departure, never `NO_MOVE`. Weak historical linear correlation reports
`UNCLEAR`; absence of a linear association cannot exclude a sampling effect.
Legacy `REAL` and `SAMPLING_ARTIFACT` codes remain only for compatibility, with
explicit descriptive `assessment` and `is_causal=false` fields. Extrapolating
outside the historical denominator range is disclosed.

Measurement coverage and economic trading volume are different. A change in
volume may itself be important; do not automatically remove it as a nuisance.
Price levels and overlapping returns can be dependent. Transforming prices into
returns alone does not demonstrate exchangeability or calibrate an alarm.

The tests cover deterministic adverse sequences, forecast replay, input
contracts and specified synthetic data generators. They do not constitute
validation of financial prediction, customer outcomes or trading performance.
