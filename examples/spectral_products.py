"""Synthetic product-specific panels; no product service or feed is contacted.

Run: python examples/spectral_products.py
Export one CLI/REST request: python examples/spectral_products.py --request seiche
The generated values demonstrate contracts, not actual institutions or markets.
"""
import argparse
import json
import random
from datetime import datetime, timedelta, timezone

from noisefloor import spectral


PROFILES = {
    "liquilens": ("spread", "bp", ["peer-a-funding-spread", "peer-b-funding-spread", "peer-c-funding-spread", "peer-d-funding-spread"]),
    "seiche": ("rate", "percent", ["funding-a", "funding-b", "funding-c", "funding-d"]),
    "undertow": ("spread", "bp", ["venue-a-spread", "venue-b-spread", "venue-c-spread", "venue-d-spread"]),
    "riptide": ("return", "fraction", ["scenario-factor-a", "scenario-factor-b", "scenario-factor-c", "scenario-factor-d"]),
    "trading_agents": ("return", "fraction", ["strategy-a", "strategy-b", "strategy-c", "strategy-d"]),
    "palimpsest": ("count", "observations", ["source-family-a", "source-family-b", "source-family-c", "source-family-d"]),
}


def request_for(product):
    """Build one explicitly synthetic, internally comparable panel."""
    kind, unit, names = PROFILES[product]
    rng = random.Random(6116)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clocks = [(start + timedelta(days=i)).isoformat() for i in range(121)]
    common = [rng.gauss(0, 1) for _ in clocks]
    series = []
    for name in names:
        level, observations = 100.0, []
        for i, (stamp, factor) in enumerate(zip(clocks, common)):
            # Stronger shared movement in the second half is constructed, not found.
            move = (0.15 if i <= 60 else 0.9) * factor + rng.gauss(0, 0.3)
            level += move
            value = move / 100 if kind == "return" else float(round(level)) if kind == "count" else level
            observations.append({"observed_at": stamp, "available_at": stamp, "value": value})
        series.append({"id": name, "kind": kind, "unit": unit, "group": product,
                       "source": {"id": "synthetic:" + product, "rights": "permitted"},
                       "max_age_seconds": 86400, "max_gap_seconds": 86400,
                       "observations": observations})
    return {"as_of": clocks[-1], "series": series,
            "policy": {"window_points": 60, "step_points": 60, "max_windows": 2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", choices=tuple(PROFILES))
    args = parser.parse_args()
    if args.request:
        print(json.dumps(request_for(args.request)))
        return
    summaries = {}
    for product in PROFILES:
        report = spectral.assess(**request_for(product))
        latest = report["latest"]
        summaries[product] = {"status": report["status"],
                              "leading_variance_share": latest["leading_variance_share"],
                              "effective_rank": latest["effective_rank"],
                              "reference_outliers": latest["reference"]["above_upper_count"],
                              "dynamics": report["dynamics"]}
    print(json.dumps({"data_status": "synthetic", "products": summaries,
                      "native_product_deployment_verified": False}, indent=2))


if __name__ == "__main__":
    main()
