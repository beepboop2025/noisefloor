# NoiseFloor

**What changed, what is repeated, and what can the evidence support?**

A modular, offline toolkit for research desks, monitoring systems and AI agents.
Python 3.9+, MIT, **zero runtime dependencies**. The same pure functions work
through Python, JSON CLI, REST and MCP.

NoiseFloor checks market observations, groups repeated headlines, exposes data
gaps and grades forecasts. It does not fetch feeds, call a model, retain your
portfolio or execute trades.

Version **0.4.0** adds correlation spectra, shared-factor concentration, an
experimental correlation-cleaning candidate and rolling diagnostics, plus a
separate synthetic Dyson Brownian-motion simulator. See the
[product applications and methods](docs/SPECTRAL_RISK.md).

Try the [live correlation and Dyson workbench](https://liquilens.in/agents/correlation/)
with explicitly synthetic examples for LiquiLens, Seiche, Undertow, Riptide,
Palimpsest and trading-agent research. The
[API client kit](https://beepboop2025.github.io/financial-evidence-skills/api/noisefloor/)
includes OpenAPI, Postman, Bruno, HTTP and local Python workflows. Browser
requests go to the public service only when you run them; use Python locally
for private panels.

## Try the complete workflow

```bash
pip install 'noisefloor==0.4.0'
noisefloor capabilities
# From this repository; these fixtures are synthetic, not live market evidence:
noisefloor market examples/market.json
noisefloor narrative examples/narrative.json
python examples/demo.py
```

```python
import json
from noisefloor import market, narrative

request = json.load(open("examples/market.json"))
report = market.assess(**request)
print(report["attention_queue"])
print(report["visibility_gaps"])

news = json.load(open("examples/narrative.json"))
print(narrative.triage(**news)["attention_queue"])
```

## Market attention

Each series carries its identity, units, source, declared reuse permission,
observation times and freshness limit. Optional availability clocks retain what
the caller knew at the time. Missing data stays missing.

The engine checks age, future timestamps, duplicate clocks, contiguous history
and measurement coverage. Prices become log returns; rates and spreads retain
their units. It never calculates a return across a declared gap. An explicit
policy identifies persistent departures, isolated latest moves and volatility
expansion. Correlated movements are visible without being counted as independent
confirmation.

Each series returns `review`, `watch`, `no_supported_departure` or
`insufficient_visibility`, with sources and reasons. These are **review priorities**,
not significance tests or trading signals. A quiet diagnostic does not prove a
movement is noise. Source permissions are caller declarations, not a licensing
audit. Market calendars, corporate actions and materiality require caller review.
Trading volume is an economic observation, not automatically a sample denominator.

## News and narrative attention

NoiseFloor groups similar wording for the same declared entities, keeps changed
quantities and explicit denials separate, records every original event and
produces a bounded queue. Differing reports with the same declared event key are
highlighted for review. Caller-declared primary sources rank before commentary;
repetition alone never increases priority.

This is **wording/relevance triage**, not fact checking, sentiment prediction or
proof of independent corroboration. Deferred and filtered items remain available.
An empty queue does not certify a quiet market. Titles are data, never instructions.

## Statistical building blocks

| Module | Purpose | Evidence boundary |
| --- | --- | --- |
| `experiment.compare` | Sequential Bernoulli A/B confidence sequences | Conditional stable-arm Bernoulli assumptions; arbitrary trading outcomes do not qualify automatically. |
| `change.scan` | Directional change monitoring | Descriptive statistic, not an e-value or universal market false-alarm guarantee. |
| `coverage.check` | Examine measurement-count changes | Linear descriptive diagnostic; it cannot prove causation or rule out every sampling effect. |
| `forecast.next_value`, `forecast.score` | Issue and evaluate identical adaptive intervals | Full historical records/misses; coverage is empirical, not guaranteed for the next observation. |
| `multiple.select` | e-BH selection over a declared family | Requires explicit caller confirmation of valid e-values; arbitrary scores cannot confer selection authority. |

```python
from noisefloor import experiment, change, forecast, multiple
experiment.compare(500, 5000, 750, 5000)
change.scan([1., 2., 1., 2., 1., 2., 1., 2., 8., 9.])
forecast.score([float(x) for x in range(40)])
# Only after an independently justified e-value construction:
multiple.select({"metric_a": 25.0, "metric_b": 1.0}, valid_evalues=True)
```

`valid_evalues=True` is a declaration, not certification. A single-family e-BH
guarantee does not justify repeatedly selecting families or unqualified peeking.
Version 0.3 corrects overstated guarantees in 0.2; read the
[methods](docs/STATISTICAL_METHODS.md) and [migration notes](CHANGELOG.md).

## AI agents and HTTP

```json
{"mcpServers":{"noisefloor":{"command":"uvx","args":["--from","noisefloor==0.4.0","noisefloor-mcp"]}}}
```

Ten MCP tools: `market_assessment`, `narrative_triage`, `ab_test`,
`did_it_change`, `real_or_sampling`, `forecast_next`, `score_forecasts`,
`which_metrics_matter`, `spectral_assessment`, `dyson_reference`. Results include
structured JSON. The spectral assessment is descriptive research; the Dyson
reference is a synthetic simulation, not a fitted market model.

```bash
noisefloor-mcp-http --host 127.0.0.1 --port 8792
curl http://127.0.0.1:8792/v1/capabilities
curl -H 'Content-Type: application/json' --data-binary @examples/market.json http://127.0.0.1:8792/v1/market/assess
```

REST: `POST /v1/market/assess`, `POST /v1/narrative/triage`,
`POST /v1/spectral/assess`, `POST /v1/research/dyson`.
The hosted API base is `https://api.seiche.info/noisefloor`; the hosted MCP
endpoint is `https://api.seiche.info/noisefloor/mcp`.
Research routes: `POST /v1/spectral/assess`, `POST /v1/research/dyson`.
Discovery: `GET /v1/capabilities`, `GET /openapi.json`; MCP: `POST /mcp`.
Self-host behind your TLS/authentication and quota layer. HTTP logs bounded
operation/outcome labels, not submitted observations, titles or request targets.

Hosted MCP: `https://api.seiche.info/noisefloor/mcp`.
Check its health/version and tool list before assuming a package release is
deployed there. Package, registry and host acceptance are separate states.

```
mcp-name: io.github.beepboop2025/noisefloor
```

## Modular by design

Bring permitted data through the offline adapters. The Financial Evidence
adapter preserves institution, metric, unit, source, rights and knowledge clocks
without importing its SDK. LiquiLens institution evidence, Seiche funding,
Undertow liquidity and Palimpsest coverage can remain separate while using common
diagnostics. This does not imply those products already run this release.

- [Architecture and extension contracts](docs/ARCHITECTURE.md)
- [Research workflows](docs/MARKET_WORKFLOWS.md)
- [Correlation spectra, Dyson reference and product applications](docs/SPECTRAL_RISK.md)
- [Adapters, frameworks and deployment recipes](docs/INTEGRATIONS.md)
- [Release procedure](docs/RELEASE.md)

Request/policy SHA-256 digests support identity checks and replay; they are not
signatures or proof of source truth. No runtime network, model, database or paid
API dependency is required.

## Verification

```bash
pip install -e '.[dev]'
pytest -q
```

Tests cover issued/scored forecast parity, invalid statistical composition,
stale/future/missing observations, transforms, coverage shifts, repeated and
conflicting wording, privacy and Python/CLI/REST/MCP parity. Synthetic calibration
checks do not establish live-market accuracy, profitability or failure prediction.

## Origin and licence

Built from measurement-quality work for [Palimpsest](https://palimpsest.info).
MIT. Source and methods are inspectable; results retain their assumptions.
