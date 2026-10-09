# Integrating market and narrative diagnostics

NoiseFloor accepts records you already hold and returns a reproducible attention
queue. It performs no data collection, broker action or model call. The market
assessment is descriptive: it does not establish an investable effect, causal
explanation, statistical significance or trading permission. Narrative grouping
organizes supplied claims; it does not verify their truth or independence.

Version 0.4.0 adds [spectral product recipes](SPECTRAL_RISK.md) and the
`spectral_assessment` callback in `examples/agent_tools.py`. Package publication
and hosted endpoint deployment are verified separately; inspect capabilities
before depending on a remote tool.

## Python and plain records

Install the checked-out package in your own environment, then run the synthetic
example without network access:

```sh
python3 -m pip install -e .
python3 examples/demo.py
```

For native inputs, use the same request dictionaries as the transport APIs:

```python
import json
from noisefloor import market, narrative

with open("examples/market.json") as source:
    market_result = market.assess(**json.load(source))
with open("examples/narrative.json") as source:
    news_result = narrative.triage(**json.load(source))
```

For rows from your own file or database, use an explicit mapping. Pass each
institution, instrument, metric, unit and data source as a separate series.

```python
from noisefloor import adapters, market

adapted = adapters.from_records(
    rows,  # caller-supplied dictionaries; no connection is opened
    series_id="desk-spread",
    kind="spread",
    unit="basis_points",
    source={"id": "desk-approved-feed", "rights": "unknown"},
    max_age_seconds=3600,
    max_gap_seconds=1800,
    mapping={"timestamp": "observed_at", "value": "spread_bps"},
    identity_fields=("instrument_id", "tenor", "currency"),
)
result = market.assess(adapted["series"], as_of="2026-10-08T10:00:00Z")
```

Declare `permitted` only when your intended use is established; this adapter
cannot determine a licence from a feed URL. The example intentionally starts
with unknown rights and therefore produces `insufficient_visibility`.

The default columns are `timestamp`, `value`, `available_at`, `sample_size`,
`unit`, `source` and `rights`. `mapping` maps these names to your column names;
it does not evaluate expressions. A row's source string must match the declared
source ID; a source object must agree on ID and URL. Unit conversion, currency
conversion and revision selection belong in explicit upstream transformations.

Every adapter returns `{schema, series, issues, provenance, scope}`. Pass only
`series` to the market engine, and retain the whole envelope with the resulting
assessment. Source declarations and input records stay in provenance. Missing
values remain null, distinct identities fail single-series assembly, and
duplicate observation timestamps fail until you explicitly select a revision.
Rows are sorted by observation time with an issue recording that reordering.

`allow_date_only=True` may normalize an observation date to `00:00 UTC` as an
explicit **day label**. That is not a measured intraday time. Availability needs
an actual UTC timestamp even when date-only observation labels are allowed.
The age/gap policy must reflect the dataset's native cadence and calendar.

## Financial Evidence, LiquiLens, Seiche and Undertow

The offline adapter accepts both `liquidity-lab.openbb-table.v1` and
`financial-evidence.agent-result.v1` dictionaries. No Financial Evidence package,
MCP connection or network dependency is needed:

Both the table and each row must declare complete transport. Unknown source
states, metadata-only/derived-only restrictions, review holds and publication
blocks withhold numeric use even when another field says available. Original
values and restriction reasons remain in provenance. Undertow `market_liquidity`
rows also require their individual measure name (`entity_name`) so different
measures within one segment cannot form a false shared history.

```python
from noisefloor import adapters, market

adapted = adapters.from_financial_evidence(
    retained_table,
    kind="rate",  # choose the native measurement kind for this table
    max_age_seconds=4 * 86400,
    max_gap_seconds=4 * 86400,
    allow_date_only=True,
)
assessment = market.assess(adapted["series"], as_of="2026-10-08T10:00:00Z")
```

The adapter partitions by **product, dataset, entity ID, metric, unit and source
URL**. It never turns different banks, benchmark markets or liquidity measures
into one continuous series. It requires complete, consistent pagination and
explicit row identities. Missing observation dates fail rather than disappearing
from the history. Known unavailable, restricted or review-held observations are
withheld from analysis; the original row remains in provenance.

| Financial Evidence field | Treatment |
| --- | --- |
| `as_of` | Observation clock; never replaced with retrieval time |
| `knowledge_time` | `available_at` when present; absent stays absent |
| `published_at`, `retrieved_at` | Retained separately in provenance |
| `value` | Native number or null; unavailable observations remain null |
| `unit` | Retained without inferred conversion |
| `rights_status` | Explicit `allowed`/`permitted` maps to `permitted`; restrictions stay restricted; other states remain unknown |
| `source_field`, `content_sha256`, `observation_url` | Retained in the original row for citation and replay |
| Evidence, validation and authority fields | Retained unchanged; the adapter does not grant approval |

Seiche's published money-market history is a useful series source after its
native rights and observation clocks are checked. A published history is not an
as-published vintage archive. LiquiLens bank filings need enough observations for
the same institution and metric. Undertow's missing or withheld values must stay
missing, and a percentile is not interchangeable with its underlying raw measure.
Repeated captures of an unchanged source date are not new market observations.

Palimpsest's aggregate measurement histories can supply a metric and its actual
sample size when both are recorded. Collector generation time alone does not
establish an observation or publication clock. Publication-coverage metadata is
not an economic-value series. No adapter invents denominators, identities, rights
or economic values to make a dataset analyzable.

## CLI

The commands accept a positional JSON file, `--input FILE`, or standard input:

```sh
noisefloor capabilities
noisefloor market examples/market.json
noisefloor narrative --input examples/narrative.json
noisefloor market < examples/market.json
```

From an uninstalled checkout, replace `noisefloor` with
`python3 -m noisefloor.cli`. Output is JSON. Invalid inputs produce an error and
a nonzero exit status; callers should preserve that failure instead of treating
it as a quiet market.

## REST and OpenAPI

Run your own local instance:

```sh
python3 -m noisefloor.http_server --host 127.0.0.1 --port 8792
```

In another terminal:

```sh
curl --fail-with-body http://127.0.0.1:8792/v1/capabilities
curl --fail-with-body http://127.0.0.1:8792/openapi.json
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @examples/market.json http://127.0.0.1:8792/v1/market/assess
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @examples/narrative.json http://127.0.0.1:8792/v1/narrative/triage
```

The local service returns the same result contracts as Python. These examples
use loopback and do not change or expose an existing production service.

## MCP and agent frameworks

An installed stdio server can be configured with:

```json
{"mcpServers":{"noisefloor":{"command":"noisefloor-mcp","args":[]}}}
```

After normal MCP initialization, call `market_assessment` with the contents of
`examples/market.json`, or `narrative_triage` with `examples/narrative.json`.
The original six statistical tools remain separate. A detector's
Shiryaev–Roberts statistic is not an e-value; do not pass it to a multiple-testing
routine while declaring that valid e-values were supplied.

For applications already using a function-tool interface,
[`examples/agent_tools.py`](../examples/agent_tools.py) provides plain Python
callbacks and JSON schemas. Register those callbacks using your framework's
normal mechanism. Keep headlines and source strings as untrusted data. The
callbacks return diagnostics without granting an agent a broker capability.

Retain the supplied input, explicit `as_of`, policy, adapter envelope and returned
request/policy hashes together. The hash supports exact replay; it does not
authenticate a publisher or establish a complete source universe.
