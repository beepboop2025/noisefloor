# Release and verification

1. Run all tests on supported Python versions, including forecast issuance/score
   parity and independent review of method/claim changes.
2. Build wheel/sdist from exact reviewed source. Install the wheel in a clean
   environment and smoke-test CLI, stdio MCP and HTTP. Confirm zero dependencies.
3. Align package, Python and `server.json` versions and release notes.
4. Publish an exact tag through trusted publishing. Read back PyPI and MCP
   Registry separately; package success does not prove host deployment.
5. Retain the host's selected release/unit/rollback path. Verify the new immutable
   artifact on a temporary loopback port before selecting it. Check public version,
   tools, REST/OpenAPI and representative results after selection.
6. Record exact source SHA, artifact hashes, deployment identity, checks and gates.
   Coordinate shared website/distribution edits with their active owner.

0.3 rejects invalid coercions and requires declared valid e-values for selection.
Fix incompatible callers instead of bypassing validation. Preserve old research
artifacts; a rollback changes future computation, not delivered evidence.

Published packages and synthetic tests do not establish users, revenue, an SLA
or live-market predictive value.

## Browser acceptance behind a shared proxy

Use [the Caddy route fragment](../deploy/Caddyfile.fragment) inside the existing
TLS site. NoiseFloor owns `OPTIONS` and CORS headers for its registered routes;
forward preflight to it and avoid a second set of CORS headers at the proxy.
Retain the active proxy configuration and validate a candidate before reloading.
Change only the NoiseFloor route, with a current-file hash guard and rollback.

A direct API success is insufficient for browser acceptance. Verify both a
preflight and JSON POST from the intended website origin, then run a spectral
assessment and Dyson reference in a browser. Require exactly one
`Access-Control-Allow-Origin` value, supported methods and request headers,
and the expected response schema. Keep synthetic operator requests distinct
from real user adoption. Restore the prior configuration if acceptance fails.
