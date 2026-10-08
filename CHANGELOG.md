# Changelog

## 0.3.1

- Update the pinned PyPI publisher to support Core Metadata 2.5 from current
  package builds. Metadata validation and trusted publishing remain enabled.
- This is the first published release of the 0.3 feature set. The immutable
  0.3.0 tag remains recorded; its upload failed before any PyPI publication.

## 0.3.0

### New

- Source-aware market attention assessments: observation/availability clocks,
  freshness, missingness, price-to-return transformation, measurement coverage,
  correlated-movement diagnostics and configurable review priorities.
- Narrative triage retaining original events, source families, changed quantities,
  denials, potential conflicts and deferred items.
- Offline adapters and Financial Evidence integration without its SDK dependency.
- Shared Python/CLI/REST/MCP contracts, capabilities/OpenAPI and structured results.

### Statistical corrections and migration

0.2 made general claims its implementation did not support. Upgrade before
relying on its outputs for market research.

- `change.scan()['evidence']` remains a raw-statistic compatibility alias, **not
  an e-value**. Universal sequential false-alarm claims have been removed.
- `multiple.select` defaults to `valid_evalues=False`, with no selected findings.
  Explicit opt-in requires independently justified e-values. Never opt in for raw
  change statistics. Single-family e-BH is not a repeated-monitoring guarantee.
- `forecast.score` evaluates exactly the forecasts issued on historical prefixes
  and publishes all evaluation records/misses. Coverage is empirical.
- `coverage.check` rejects mismatched arrays, recognizes constant-history jumps,
  and does not treat weak correlation as proof of a real move.
- Finite values, true integer Bernoulli counts, denominators and parameters are
  validated without silent coercion. Statistical assumptions are explicit.
- HTTP accepts declared routes only. Malformed requests, duplicate JSON keys,
  nonfinite values, unknown fields and oversized inputs are rejected.

The six existing MCP tool names remain; their validity contracts are more
restrictive where necessary for correctness. New market/narrative tools support
research attention, not broker execution, feed licensing or alpha certification.
Package publication, host deployment and adoption require separate evidence.
