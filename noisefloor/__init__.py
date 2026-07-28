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

Every method is anytime-valid or distribution-free, which is the property that
survives contact with how people really use dashboards: looking whenever they
feel like it, and stopping when they see what they want.

ZERO DEPENDENCIES. Standard library only, no numpy, no scipy. It installs
anywhere Python does — lambda functions, edge runtimes, locked-down build
images — and every result is deterministic and reproducible.
"""
from __future__ import annotations

from . import change, coverage, experiment, forecast, multiple

__version__ = "0.1.0"
__all__ = ["change", "coverage", "experiment", "forecast", "multiple"]
