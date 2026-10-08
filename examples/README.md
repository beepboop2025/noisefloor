# Offline synthetic examples

Every value, market, company and headline in this directory is **synthetic**.
The fixtures describe fixed dates, so reruns do not drift with the current clock.
`rights: permitted` on the generated series describes these synthetic fixtures,
not permission for an external market-data feed. `example.invalid` URLs are
provenance placeholders and are never fetched.

Run from the repository checkout:

```sh
PYTHONPATH=. python3 examples/demo.py
```

The program prints an adapter result summary, market diagnostics and a narrative
queue as JSON. It makes no network calls or file writes. `market.json` and
`narrative.json` are also ready-to-use inputs for the Python, CLI and REST APIs.

The market example includes a persistent rate movement, a quote-count movement
with reduced sampling coverage, and a source with unknown usage rights. The
last case should remain `insufficient_visibility`. No result is a trading signal
or a claim of statistical significance.

The narrative example includes repeated wording from a declared syndication
family, a changed numerical claim, a denial, an unmapped entity, an event outside
the selected entity, and a future publication. Repetition does not establish
corroboration. Distinct quantities and denials must remain visible for review.

`agent_tools.py` exposes framework-neutral callbacks and their owned JSON input
schemas. Register a callback with the framework you already use; these examples
do not install an SDK, invoke a model, start a scheduler, or grant trading access.
