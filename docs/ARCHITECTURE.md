# Modular architecture

The same engine serves a private research notebook, dashboard or AI agent without
changing the meaning of a result.

```mermaid
flowchart LR
 S[Caller-owned sources] --> A[Offline adapters]
 A --> C[Observations and source clocks]
 C --> Q[Coverage and eligibility checks]
 Q --> M[Market diagnostics]
 C --> N[Narrative wording triage]
 M --> R[Evidence and review priorities]
 N --> R
 R --> P[Python or JSON CLI]
 R --> H[REST or MCP]
 H --> U[Human or agent research workflow]
```

| Module | Owns | Boundary |
| --- | --- | --- |
| `adapters` | Field mapping, identities, source metadata | No fetching, rights inference or silent revision selection |
| `schemas` | Bounded strict JSON contracts | Shared across interfaces |
| `market` | Time/rights/coverage gates, transforms, review policy | Descriptive, no execution |
| `narrative` | Repetition, entity focus, source ordering | No truth, independence or impact claim |
| Statistical modules | Individual methods and assumptions | Raw scores are not e-values |
| `mcp_server` | Discovery and shared dispatch | Same result as Python |
| `http_server` | Framing, bounded requests and privacy-safe logging | No persistent portfolio or credentials |
| `cli` | JSON input/output | Offline, explicit errors |

The caller supplies `as_of`; computation never consults the wall clock. Source
observations and availability clocks stay separate. Unknowns remain unknown.
Exact request/effective-policy digests travel with results. Revisions require an
explicit upstream selection policy. Metadata declarations are not attestations.

Requests are limited to 32 market series, 2,048 points each, 500 headlines, a 2 MiB
transport body and bounded JSON-RPC batches. Market diagnostics use a configured
contiguous window of at most 288 transformed points. Excess inputs are rejected.
HTTP admits four simultaneous connections per process, including idle keepalive
connections; excess connections receive a retryable 503. TLS, per-client quotas
and tenant authentication remain responsibilities of the hosting layer.

Health and capabilities include a SHA-256 fingerprint of installed Python source
files, captured once per process. Compare it with the release's `source.json`.
Immutable release directories are required for this operational identity check;
the fingerprint is not a signature or a full runtime attestation.

Add sources through offline adapters, statistics as separate pure functions with
assumption contracts, and tools through shared dispatch. Keep collection/auth,
LLM interpretation, persistence and order routing outside the core.

Large universes can be partitioned by instrument/source/time with a caller-owned
coverage inventory. Partitioning does not establish multiple-testing validity.
A managed multi-tenant deployment additionally needs identity, quotas, retention,
resource isolation and operational acceptance; those belong at the host boundary.
