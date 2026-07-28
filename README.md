# noisefloor

**Is this number real, or is it noise?**

Zero dependencies. Standard library only. Works as a Python package or as an MCP server.

---

## The problem

You ship a change and check the A/B test. Then you check again an hour later. Then tomorrow. You stop when it looks like a win.

That procedure lies to you, and not by a little:

| 400 A/A tests, both arms identical, peeked every 20 observations | False winners |
|---|---|
| Two-proportion z-test | **40.2%** |
| noisefloor | **0.0%** |

Both arms had the *same* 10% conversion rate, so every "winner" was false. A test tuned to be wrong 5% of the time was wrong 40% of the time, purely because someone looked more than once.

This isn't a discipline problem. Peeking is the rational thing to do when a bad variant is costing money. It's a maths problem, and it has a solution.

Reproduce the table with `pytest tests/test_calibration.py`.

## Install

```bash
pip install noisefloor
```

## Use

```python
from noisefloor import experiment

experiment.compare(a_successes=500, a_total=5000,
                   b_successes=750, b_total=5000)
# {'decided': True, 'winner': 'B',
#  'reading': "B wins: its interval sits entirely above A's",
#  'A': {'rate': 0.1, 'interval': [0.08366, 0.11809]},
#  'B': {'rate': 0.15, 'interval': [0.13049, 0.17101]}, ...}
```

Run it after every single observation if you like. The guarantee holds at every sample size simultaneously, so stopping early, stopping late, or stopping because your manager walked past all cost you nothing.

### The other four checks

```python
from noisefloor import change, coverage, forecast, multiple

# Did this metric actually change? Catches collapses as well as spikes.
change.scan(daily_signups)
# {'state': 'changed', 'direction': 'down',
#  'reading': 'CHANGED (down), 8.5x its usual spread', ...}

# Did the metric move, or did my sample size move?
coverage.check(conversion_rate, sessions_per_day)
# {'verdict': 'SAMPLING_ARTIFACT',
#  'reading': 'your sample size changed by -41% at the same time and it tracks
#              this metric historically (correlation 0.70)...'}

# What should the next reading be?
forecast.next_value(latency_p95)

# How good have these forecasts actually been?
forecast.score(latency_p95)
# {'empirical_coverage': 0.803, 'calibrated': True, 'n_misses': 197,
#  'worst_misses': [...]}   # misses are always published

# I watch 40 metrics. Which genuinely stand out?
multiple.select({'signups': 3.2, 'latency': 812.0, 'errors': 1.1, ...})
# {'selected': ['latency'],
#  'reading': '1 of 40 metrics worth looking at: latency'}
```

## As an MCP server

Assistants read metrics constantly and assert significance the way people do — by eyeballing a change and calling it. This gives them a way to check.

```json
{
  "mcpServers": {
    "noisefloor": { "command": "uvx", "args": ["--from", "noisefloor", "noisefloor-mcp"] }
  }
}
```

```
mcp-name: io.github.beepboop2025/noisefloor
```

Six tools: `ab_test`, `did_it_change`, `real_or_sampling`, `forecast_next`, `score_forecasts`, `which_metrics_matter`.

## Why these methods

Every check is **anytime-valid** or **distribution-free** — the two properties that survive contact with how dashboards are really used: looked at whenever someone feels like it, and stopped when they see what they want.

- **`experiment`** — beta-binomial mixture test martingale. A mixture of likelihood ratios is a non-negative martingale under the null, so Ville's inequality bounds the chance it *ever* crosses `1/alpha`. Exact for Bernoulli outcomes rather than a worst-case bound, which is where the power comes from. ([theory](https://arxiv.org/abs/2210.01948), [safe testing](https://arxiv.org/abs/1906.07801))
- **`change`** — conformal Shiryaev-Roberts detector. Rank-based p-values, no distributional assumption, with a stated average time between false alarms. Two-sided by default, because half of what goes wrong is a number going to zero. ([nonexchangeable conformal](https://arxiv.org/abs/2202.13415))
- **`coverage`** — conditions the metric on its own denominator. Almost every metric worth watching is a rate over a sample you don't control.
- **`forecast`** — adaptive conformal intervals, valid under arbitrary distribution shift, graded by the Weighted Interval Score, a *proper* rule so the scoreboard can't be gamed by hedging. ([ACI](https://arxiv.org/abs/2106.00170), [decaying steps](https://arxiv.org/abs/2402.01139), [WIS](https://arxiv.org/abs/2005.12881))
- **`multiple`** — e-Benjamini-Hochberg. Controls false discoveries across all your metrics at once, under *arbitrary dependence* — which matters, because real metrics move together. ([e-BH](https://arxiv.org/abs/2009.02824), [merging](https://arxiv.org/abs/1912.06116))

## What it costs

Anytime validity isn't free. At any fixed sample size the interval is wider than a one-look interval, so calling the same effect takes roughly twice the data — 10% vs 13% resolves at a median of about 14,300 observations here, against roughly 7,000 for a correctly run one-look test.

You're buying the right to stop whenever you want. For most teams that's a bargain, because the realistic alternative isn't a clean one-look test. It's a one-look test being peeked at, which is the 40% column above.

## Design

Nothing here returns a number it can't stand behind. Not enough history returns `NOT_ENOUGH_HISTORY`, not a confident zero. A metric that moved with its own sample size returns `SAMPLING_ARTIFACT`, not a finding. Forecast misses are published in full and there's no flag to hide them.

Every result carries its `method` and, where one exists, its `guarantee`. Quote them — the point is that the claim can be backed rather than asserted.

Zero dependencies means it installs in Lambda, edge runtimes and locked-down build images where adding scipy is a procurement conversation. Every result is deterministic: same input, same answer, forever, with no RNG anywhere.

## Provenance

These engines were built for [Palimpsest](https://palimpsest.info), a public-good censorship observatory, where publishing a number you can't defend is the whole failure mode. The sampling-artifact check exists because a censorship index there fell 60.3 → 55.6 while the measurements underneath it fell 353,676 → 208,933. The index hadn't moved. The instrument had.

## Licence

MIT.
