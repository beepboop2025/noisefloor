"""Framework-neutral tool callbacks; no agent framework dependency required.

Register these functions with your framework's normal function-tool facility.
Input/output are plain JSON-compatible dictionaries; no execution permission is
conveyed by a diagnostic result. No prompt, URL, source text, or headline is run.
"""
from noisefloor import market, narrative, spectral
from noisefloor.schemas import MARKET_REQUEST, NARRATIVE_REQUEST, SPECTRAL_REQUEST, validate


def market_assessment(arguments: dict) -> dict:
    """Assess caller-supplied series, preserving unknown and blocked states."""
    validate(arguments, MARKET_REQUEST)
    return market.assess(**arguments)


def narrative_triage(arguments: dict) -> dict:
    """Group repeated wording and rank the supplied events for attention."""
    validate(arguments, NARRATIVE_REQUEST)
    return narrative.triage(**arguments)


def spectral_assessment(arguments: dict) -> dict:
    """Review shared factors and noise-reference bounds; no execution authority."""
    validate(arguments, SPECTRAL_REQUEST)
    return spectral.assess(**arguments)


TOOLS = [
    {"name": "market_assessment", "description": market_assessment.__doc__,
     "input_schema": MARKET_REQUEST, "callback": market_assessment},
    {"name": "narrative_triage", "description": narrative_triage.__doc__,
     "input_schema": NARRATIVE_REQUEST, "callback": narrative_triage},
    {"name": "spectral_assessment", "description": spectral_assessment.__doc__,
     "input_schema": SPECTRAL_REQUEST, "callback": spectral_assessment},
]
