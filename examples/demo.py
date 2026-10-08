#!/usr/bin/env python3
"""Run both deterministic synthetic examples, including the offline adapter.

From the checkout: PYTHONPATH=. python3 examples/demo.py
No network, broker, model SDK, credentials, or output-file writes are involved.
"""
from pathlib import Path
import json

from noisefloor import adapters, market, narrative
from noisefloor.schemas import loads


def main():
    folder = Path(__file__).resolve().parent
    market_input = loads((folder / "market.json").read_text())
    news_input = loads((folder / "narrative.json").read_text())
    first = market_input["series"][0]
    rows = [{"timestamp": point["observed_at"], "value": point["value"],
             "available_at": point["available_at"], "unit": first["unit"],
             "entity_id": "SYNTHETIC-MARKET", "metric": "synthetic-rate"}
            for point in first["observations"]]
    adapted = adapters.from_records(
        rows, series_id=first["id"], kind=first["kind"], unit=first["unit"],
        source=first["source"], group=first["group"],
        max_age_seconds=first["max_age_seconds"], max_gap_seconds=first["max_gap_seconds"])
    market_input["series"][0] = adapted["series"][0]
    output = {"input_class": "synthetic_demonstration", "network_calls": 0,
              "adapter_issues": adapted["issues"],
              "market": market.assess(**market_input),
              "narrative": narrative.triage(**news_input)}
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
