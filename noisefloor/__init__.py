"""noisefloor — is this number real, or is it noise?

Five checks that answer the questions people actually argue about in dashboard
reviews and experiment readouts:

    experiment.compare   can I call a winner on this A/B test yet?
                         (safe to check as often as you like)
    change.scan          did this metric actually change, or is it noise?
    coverage.check       did the metric move, or did my sample size move?
    forecast.next_value  what should the next reading be, and within what range?
    forecast.score       how good have these forecasts actually been?
    multiple.select      I watch 40 metrics; which ones genuinely stand out?

Methods have explicit assumptions. Market and narrative assessments are
descriptive; unqualified monitoring statistics are not valid e-values.

ZERO DEPENDENCIES. Standard library only, no numpy, no scipy. It installs
anywhere Python does — lambda functions, edge runtimes, locked-down build
images — and every result is deterministic and reproducible.
"""
from __future__ import annotations

__version__ = "0.4.0"

from . import change, coverage, experiment, forecast, multiple, market, narrative, adapters, spectral, dyson

__all__ = ["change", "coverage", "experiment", "forecast", "multiple", "market", "narrative", "adapters", "spectral", "dyson"]
