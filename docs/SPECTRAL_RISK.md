# Shared factors, sampling noise and Dyson dynamics

**Dyson Brownian motion** describes how random movement of matrix entries
produces interacting, repelling eigenvalues. A related financial application is
random-matrix analysis of sample correlations. This research capability makes
both ideas available under separate observation and simulation contracts.

## Two distinct computations

`spectral.assess` analyzes permitted caller-supplied observations. It reports
the spectrum of a sample correlation matrix, concentration, a reference noise
band, an experimental shrinkage candidate, and changes between rolling windows.

`dyson.simulate` produces a synthetic beta=1 Brownian matrix reference. Its
eigenvalues can be negative. It is **not** a covariance model, a financial
simulation calibrated to observations, or a price prediction. Financial
correlation matrices are positive semidefinite; their random reference is a
sample-covariance/Wishart model. A Dyson simulation does not calibrate the
Marchenko-Pastur edges or certify empirical risk factors.

## Product application

| Consumer | Comparable inputs | Useful review output | Required boundary |
| --- | --- | --- | --- |
| NoiseFloor | A caller-defined panel of histories | Shared modes, concentration, noise reference and rolling movement | Research diagnostics, no universal real/noise verdict |
| Trading-agent research | Aligned instrument returns or strategy return histories | Detect several proposals depending on the same exposure; compare raw and cleaned correlations | No position sizing, order authorization or profitability inference |
| LiquiLens | Same-definition institution metrics within an approved jurisdiction and reporting cadence | Common funding/market exposure versus institution-specific review | No credit score, causal contagion claim or cross-country calibration from correlations |
| Seiche | Comparable funding rates and spreads on aligned dates | Shared funding movement, concentration and rotation | Source observation dates, permitted use and locally appropriate transformations |
| Undertow / Undertow-MM | Comparable venue spreads or liquidity histories | Venues moving together; duplicated apparent liquidity diversity | Execution costs and liquidity shocks need their own validation |
| Riptide | Research factor histories admitted to a scenario study | Raw versus cleaned dependency sensitivity | A scenario experiment cannot authorize allocation; Dyson draws are not calibrated market shocks |
| Palimpsest | Consistently defined measurement/coverage counts across source families | Shared changes in coverage that can explain repeated apparent evidence | Measurement association does not establish truth, influence or independent corroboration |

Sparse quarterly institution filings must not be forward-filled into a daily
panel to manufacture sample size. Frequency, jurisdiction, units, population,
source vintages and definition changes need explicit upstream selection.
Declared measurement counts are summarized and variable/incomplete coverage is
flagged. The spectrum does not adjust for sampling changes; constant counts do
not prove constant source composition or unbiased measurement.
NarcoScope entity allegations and ScamShield verdicts are not suitable numeric
inputs by themselves; do not convert them to an unexplained financial factor.

NoiseFloor owns these numerical diagnostics. Existing Lab temporal/evidence
contracts retain authority over historical cuts; products retain their own
source admission, evaluation and review decisions. Native product deployment
is a separate release step, not implied by a working adapter or example.

## Python and Financial Evidence integration

```python
from noisefloor import adapters, spectral

# table is a complete, explicitly selected Financial Evidence table.
# Select a comparable panel and one observation type before adapting.
adapted = adapters.from_financial_evidence(
    table, kind="spread", max_age_seconds=3 * 86400,
    max_gap_seconds=4 * 86400, allow_date_only=True,
)
report = spectral.assess(
    adapted["series"], as_of=as_of,
    policy={"window_points": 60, "step_points": 20, "max_windows": 4},
)
# Retain the adapter envelope: it contains native identities and provenance.
evidence = {"adapter": adapted, "spectral": report}
```

The existing Financial Evidence adapter preserves LiquiLens, Seiche and Undertow
identity partitions, source states, rights and knowledge clocks. Date-only
normalization is explicit; midnight is a day label, not invented intraday
availability. A `SOURCE_REVIEW_HOLD`, restricted right or incomplete transport
remains unavailable. Supply local records via `adapters.from_records` for other
products. Neither adapter fetches data or infers a licence.

```sh
# From this candidate checkout, installed with pip install -e '.[dev]':
python examples/spectral_products.py
python examples/spectral_products.py --request seiche | noisefloor spectral
echo '{"dimension":6,"steps":40,"dt":0.05,"seed":6116}' | noisefloor dyson
```

All example data are synthetic. The names describe integration roles, not
observed product data. The framework-neutral callback is in
`examples/agent_tools.py`. MCP tools are `spectral_assessment` and
`dyson_reference`; REST routes are `POST /v1/spectral/assess` and
`POST /v1/research/dyson`. Discovery and OpenAPI expose the same strict schemas.

## Observation and numerical contract

1. Every requested series must pass the shared market source/freshness/knowledge
   checks. An ineligible member blocks the panel instead of disappearing.
2. Prices become log returns; supplied returns keep their values; rates,
   spreads, counts and other levels become first differences. Differencing is
   a default research transformation, not a stationarity test. Supply approved
   transformed increments as `kind="return"` when that convention is suitable.
3. Each measurement interval uses the preceding observation timestamp and its
   own timestamp. The first observation establishes the start and is not a
   sample. Both endpoints must match across members. For supplied returns,
   this assumes each value covers the immediately preceding declared interval.
4. Missing values or declared calendar gaps cut the contiguous history. All
   members use one common interval suffix. Nothing is interpolated or filled.
   Latest mismatches block rather than silently reverting to an earlier panel.
5. Each window is centered and standardized separately. Constant or numerically
   unresolved series produce an unavailable window. Unit scaling occurs before
   squaring, preserving very small and very large units without overflow.
6. `C = Z^T Z/(T-1)` is a Gram correlation matrix. The bounded symmetric Jacobi
   eigensolver returns descending eigenvalues and orthonormal eigenvectors,
   or raises on nonconvergence. No runtime numerical dependency is added.

Requests allow 2–32 members, 20–512 transformed observations per window, and at
most eight windows. Input histories retain the existing 2,048-point bound.
The iid reference requires `T > N`. Restricted, empty, stale, undersampled and
constant panels stay explicit. There is no implicit smaller-universe fallback.

The unit-variance reference edges use
`lambda_minus/plus = (1 -/+ sqrt(q))^2`, with `q = N/(T-1)` as the stated
degrees-of-freedom convention. These are **asymptotic support edges**, not
finite-sample confidence bounds. The implementation does not estimate effective
sample size, fit a residual-noise variance or assume stationarity was proven.
Lag-one correlations are descriptive warnings for review; their absence does
not establish independence. Heavy tails, serial dependence, common volatility,
overlapping observations and adaptive window selection need separate treatment.

Concentration is `lambda_max / N`; participation ratio is
`N^2 / sum(lambda_i^2)`; entropy effective rank is
`exp(-sum(p_i log(p_i)))`, where `p_i = lambda_i/N`. They describe standardized
dependence, not portfolio risk in currency or independent economic factor count.
Loadings have arbitrary sign. The reported sign is fixed for display; squared
overlap is sign invariant. Nearly degenerate leading modes have no unique
direction, so their overlap is unavailable rather than a reported rotation.

The shrinkage candidate replaces only in-band eigenvalues with their mean,
preserving the trace before reconstructing and restoring a unit diagonal.
Lower and upper outliers survive that intermediate step. Diagonal
normalization can change the spectrum again. This candidate remains PSD but
may be singular, does not guarantee conditioning or improved risk estimates,
and must not be inverted blindly. No return forecast or portfolio allocation
is computed.

Rolling matrices use identical member sets and past-only sample slices. Windows
can overlap, so their movements are not independent tests. Historical slices
of a currently supplied vintage do not establish what was published or known
at the time. Use verified knowledge-time cuts before point-in-time evaluation.

## Dyson reference

For a real symmetric `N x N` matrix beginning at zero, independent increments
have variance `dt/N` off the diagonal and `2*dt/N` on the diagonal. Diagonalizing
each sampled matrix realizes the beta=1 eigenvalue process on that time grid.
Away from collisions its normalization corresponds to
`d lambda_i = sqrt(2/N) dB_i + (1/N) sum_{j!=i} 1/(lambda_i-lambda_j) dt`.
The implementation simulates matrix increments, not a singular Euler drift.
It has no confining/mean-reversion term and no stationary GOE claim. All initial
eigenvalues coincide at time zero; sampled later paths are sorted by rank.

The simulator admits dimensions 2–16, at most 100 steps, and explicit bounded
`dt` and seed. It uses a local RNG and cannot alter the caller's global RNG.
Record the Python runtime and implementation fingerprint with the seed for replay.

## Evaluation before product promotion

- Keep an independently selected, permitted historical panel with known source
  and availability clocks. Lock window, transform and missingness rules first.
- Compare raw sample correlations, this candidate, a diagonal model and a
  conventional shrinkage baseline on held-out periods. Report covariance loss,
  realized risk-estimation error, instability, missingness and negative results.
- If a paper strategy consumes the output, compare after fees, spread, impact
  and turnover with the same risk policy. Account for strategy selection and
  repeated testing. Numeric improvement does not by itself confer live routing.
- Bootstrap or simulate an appropriate dependent null before publishing
  calibrated alarm thresholds; do not use ideal MP edges as a false-alarm rate.
- Review numerical correctness, source rights and each native product's
  admission gate. Verify exact package, registry and hosted identities separately.

## Primary references

- [Dyson, A Brownian-Motion Model for the Eigenvalues of a Random Matrix (1962)](https://doi.org/10.1063/1.1703862): the matrix/eigenvalue Brownian relationship.
- [Laloux, Cizeau, Bouchaud and Potters, Noise Dressing of Financial Correlation Matrices (1999)](https://arxiv.org/abs/cond-mat/9810255): financial sample correlations and random-matrix noise reference.

These sources motivate the model choice. They do not validate this implementation
or the supplied product data. The integration choices and evaluation requirements
above are engineering proposals for this product family.
