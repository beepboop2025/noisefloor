"""Rank metrics; apply e-BH only to explicitly caller-confirmed valid e-values.

A valid e-value is nonnegative and has expectation at most one under its null.
Magnitude alone cannot establish this property. A Shiryaev-Roberts monitoring
statistic, z-score, raw return or language-model confidence is not an e-value.
Wang & Ramdas (2022), https://arxiv.org/abs/2009.02824.
"""
from __future__ import annotations

from ._validation import number, probability, series

DEFAULT_ALPHA = 0.10


def merge(evidence: list[float]) -> float:
    """Arithmetic mean; e-value validity is conditional on valid input e-values.

    This function performs arithmetic only; it does not certify its inputs.
    """
    values = series(evidence, "evidence")
    if any(value < 0 for value in values):
        raise ValueError("evidence must be nonnegative")
    return number(sum(value / len(values) for value in values), "merged evidence") if values else 1.0


def select(evidence: dict[str, float], *, alpha: float = DEFAULT_ALPHA,
           valid_evalues: bool = False) -> dict:
    """Rank inputs, optionally applying conditional one-board e-BH selection.

    ``valid_evalues=True`` is a caller assertion of a statistical contract, not
    a certification by noisefloor. Each input must have null expectation <= 1,
    with its null, sampling/selection procedure and observation window defined.
    Arbitrary dependence BETWEEN valid e-values is permitted. Repeated selection
    over time and adaptive choice of hypothesis family require further methods.
    """
    alpha = probability(alpha, "alpha")
    if not isinstance(valid_evalues, bool):
        raise ValueError("valid_evalues must be a boolean")
    if not isinstance(evidence, dict) or any(not isinstance(name, str) or not name for name in evidence):
        raise ValueError("evidence must be a mapping with non-empty string names")
    values = {name: number(value, f"evidence[{name}]") for name, value in evidence.items()}
    if any(value < 0 for value in values.values()):
        raise ValueError("evidence must be nonnegative")
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    total = len(ranked)
    k = 0
    if valid_evalues:
        for index, (_, value) in enumerate(ranked, 1):
            if value >= total / (alpha * index):
                k = index
    selected = sorted(name for name, _ in ranked[:k])
    assumptions = [
        "Each input is nonnegative with null expectation at most one.",
        "Null hypotheses, observation windows and input selection are defined before selection.",
        "Raw change statistics are not e-values and cannot be substituted.",
        "The bound concerns this supplied family; repeated boards are not automatically controlled.",
        "Caller confirmation does not verify the statistical construction or provenance.",
    ]
    if not total:
        reading = "nothing is being watched"
    elif not valid_evalues:
        reading = "inputs ranked descriptively; no discoveries selected because e-value validity is unconfirmed"
    elif selected:
        reading = f"{len(selected)} of {total} hypotheses selected, conditional on the supplied e-values being valid"
    else:
        reading = f"none of {total} hypotheses selected by conditional e-BH; this does not prove no effects"
    return {
        "selected": selected,
        "alpha": alpha,
        "k": k,
        "n_watched": total,
        "threshold": number(total / (alpha * (k or 1)), "selection threshold") if total and valid_evalues else None,
        "board_evidence": merge(list(values.values())) if total and valid_evalues else None,
        "descriptive_mean": merge(list(values.values())) if total else None,
        "ranked": [{"metric": name, "evidence": value} for name, value in ranked],
        "valid_evalues": valid_evalues,
        "validity": "conditional_on_valid_evalues" if valid_evalues else "unverified_inputs",
        "guarantee": (f"If every supplied value is a valid e-value for its defined null, e-BH controls the expected false-discovery proportion of this family at {alpha:g}, under arbitrary dependence between inputs.") if valid_evalues else None,
        "assumptions": assumptions,
        "reading": reading,
        "method": "e-Benjamini-Hochberg (arXiv:2009.02824) when valid_evalues=True; descriptive ranking otherwise",
        "note": "A selection is evidence for review, not a recommendation to trade.",
    }
