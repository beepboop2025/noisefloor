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
