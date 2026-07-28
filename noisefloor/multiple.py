"""Watching many metrics at once without drowning in false alarms.

THE PROBLEM. You monitor forty metrics. Each alert is tuned to fire wrongly
about 5% of the time, which sounds fine until you notice that on any given day
you expect two false alarms across the board. People respond by muting alerts,
which is worse than having none, because now the real one is muted too.

This is the multiple-comparisons problem, and dashboards have it badly. A
per-metric guarantee is not a board guarantee, and the number a human actually
reads off the wall is the board one.

THE TOOL. e-Benjamini-Hochberg (Wang & Ramdas, JRSS-B 2022, arXiv:2009.02824).
Given a piece of evidence per metric, sort them and select the largest k whose
k-th largest value clears K/(alpha*k). This controls the false discovery rate —
the expected share of flagged metrics that are flagged wrongly — at alpha.

The property that matters here: it holds under ARBITRARY dependence between the
metrics. That is not a technicality. Real metrics move together, because they
are all downstream of the same traffic, the same deploys and the same outages.
Any procedure that needs independence is unusable on a real dashboard, and the
common alternative (Bonferroni with a log-factor penalty) is so conservative
that people turn it off.

Evidence values are also merged into one board-level number by arithmetic mean,
which stays valid under arbitrary dependence too (Vovk & Wang, Ann. Statist.
2021, arXiv:1912.06116) and is essentially the only sensible symmetric way to
combine them. A product would explode the moment two correlated metrics moved.

Standard library only, deterministic.
"""
from __future__ import annotations

DEFAULT_ALPHA = 0.10


def merge(evidence: list[float]) -> float:
    """Combine evidence from several metrics into one board-level number.

    The arithmetic mean of evidence values is still valid evidence even when the
    metrics are correlated, which is why the mean is used rather than a product.
    """
    return sum(evidence) / len(evidence) if evidence else 1.0


def select(evidence: dict[str, float], *, alpha: float = DEFAULT_ALPHA) -> dict:
    """Which metrics are genuinely worth looking at, out of everything watched.

    evidence maps metric name -> evidence value (for example the `evidence`
    field from change.scan). Returns the selected names and the bar they had
    to clear, controlling the false discovery rate at alpha.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not evidence:
        return {"selected": [], "alpha": alpha, "k": 0, "n_watched": 0,
                "threshold": None, "board_evidence": None,
                "reading": "nothing is being watched"}

    ranked = sorted(evidence.items(), key=lambda kv: kv[1], reverse=True)
    K = len(ranked)
    k = 0
    for i in range(1, K + 1):
        if ranked[i - 1][1] >= K / (alpha * i):
            k = i

    selected = sorted(name for name, _ in ranked[:k])
    board = merge(list(evidence.values()))

    if selected:
        reading = (f"{len(selected)} of {K} metrics worth looking at: "
                   + ", ".join(selected))
    else:
        reading = (f"none of {K} metrics stands out once you account for watching "
                   f"{K} of them at once")

    return {
        "selected": selected,
        "alpha": alpha,
        "k": k,
        "n_watched": K,
        "threshold": round(K / (alpha * k), 3) if k else round(K / alpha, 3),
        "board_evidence": round(board, 4),
        "reading": reading,
        "ranked": [{"metric": n, "evidence": round(v, 4)} for n, v in ranked],
        "guarantee": (
            f"among the metrics reported as worth looking at, the expected share "
            f"flagged wrongly is at most {alpha:.0%}; holds even though the metrics "
            f"move together"),
        "method": (
            "e-Benjamini-Hochberg (arXiv:2009.02824): sort the evidence, select the "
            "largest k with the k-th value >= K/(alpha*k). Valid under arbitrary "
            "dependence, with no extra penalty factor. Board-level evidence is the "
            "arithmetic mean, which stays valid for dependent inputs "
            "(arXiv:1912.06116)."),
        "note": (
            "a per-metric error rate is not a board error rate: watching K metrics "
            "at a 5% individual rate means expecting K*0.05 false alarms per round"),
    }
