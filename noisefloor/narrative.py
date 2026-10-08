"""Deterministic news novelty and relevance triage, without truth or trade claims.

Titles are untrusted input, never instructions. No URLs are fetched. Repeated
wording is grouped for attention management, not counted as corroboration.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

from .market import _hash, _time
from .schemas import NARRATIVE_REQUEST, validate

DEFAULT_POLICY = {"max_age_seconds": 86400, "max_items": 20, "similarity_threshold": 0.8}
_NEGATION = {"no", "not", "never", "denies", "denied", "without", "false", "unconfirmed"}
_UP = {"up", "rise", "rises", "rising", "rose", "raise", "raises", "raised", "increase", "increases", "increased", "gain", "gains", "upgrade", "upgrades"}
_DOWN = {"down", "fall", "falls", "falling", "fell", "cut", "cuts", "decrease", "decreases", "decreased", "loss", "losses", "downgrade", "downgrades"}
_UNITS = {"thousand", "thousands", "million", "millions", "billion", "billions", "trillion", "trillions",
          "lakh", "lakhs", "crore", "crores", "percent", "percentage", "bps", "bp", "points", "usd", "eur", "inr", "gbp"}


def _tokens(title):
    normalized = unicodedata.normalize("NFKC", title).casefold().replace("−", "-")
    return frozenset(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*%?|\w+", normalized))


def _family(source):
    if source.get("family"):
        return source["family"]
    if source.get("url"):
        try:
            host = urlsplit(source["url"]).hostname
        except ValueError:
            host = None
        if host:
            return host.casefold().removeprefix("www.")
    return source["id"]


def _signature(tokens):
    # Never collapse changed quantities or an inserted denial as harmless copy.
    return ({t for t in tokens if any(c.isdigit() for c in t)}, tokens & _NEGATION,
            bool(tokens & _UP), bool(tokens & _DOWN), tokens & _UNITS)


def _similar(a, b, threshold):
    if a["entities"] != b["entities"]:
        return False
    left_key, right_key = a["event"].get("event_key"), b["event"].get("event_key")
    if left_key is not None and right_key is not None and left_key != right_key:
        return False
    if a["signature"] != b["signature"]:
        return False
    left, right = a["tokens"], b["tokens"]
    if not left or not right:
        return a["event"]["title"] == b["event"]["title"]
    similarity = len(left & right) / len(left | right)
    return similarity >= threshold


def triage(events: list[dict], *, as_of: str, focus: list[str] | None = None,
           policy: dict | None = None) -> dict:
    """Retain every supplied event while producing a bounded attention queue.

    Source kind/family and entity assignments are caller declarations. Selection
    never establishes truth, market impact or economic importance.
    """
    request = {"events": events, "as_of": as_of}
    if focus is not None:
        request["focus"] = focus
    if policy is not None:
        request["policy"] = policy
    validate(request, NARRATIVE_REQUEST)
    ids = [e["id"] for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError("event ids must be unique")
    applied = {**DEFAULT_POLICY, **(policy or {})}
    now, wanted = _time(as_of), set(focus or [])
    classified, candidates = [], []
    for event in events:
        published = _time(event["published_at"])
        available = _time(event.get("available_at", event["published_at"]))
        if "available_at" in event and available < published:
            raise ValueError(f"{event['id']}: availability precedes publication")
        if published > now or available > now:
            status = "not_yet_available"
        elif (now - published).total_seconds() > applied["max_age_seconds"]:
            status = "stale"
        elif wanted and event["entities"] and not wanted.intersection(event["entities"]):
            status = "outside_focus"
        else:
            status = "unmapped_entity" if wanted and not event["entities"] else "candidate"
        entry = {"id": event["id"], "status": status, "cluster_id": None,
                 "published_at": event["published_at"], "available_at": event.get("available_at"),
                 "source": dict(event["source"]), "entities": list(event["entities"]),
                 "title": event["title"]}
        classified.append(entry)
        if status in ("candidate", "unmapped_entity"):
            tokens = _tokens(event["title"])
            candidates.append({"event": event, "entry": entry, "tokens": tokens,
                               "signature": _signature(tokens), "entities": frozenset(event["entities"])})
    # Stable order makes exact reruns independent of delivery order, while the
    # request hash still proves which original input bytes were represented.
    candidates.sort(key=lambda x: (x["event"]["published_at"], x["event"]["id"]))
    clusters = []
    for item in candidates:
        match = next((c for c in clusters if _similar(item, c[0], applied["similarity_threshold"])), None)
        if match is None:
            clusters.append([item])
        else:
            match.append(item)
    summaries = []
    kind_order = {"primary": 0, "reporting": 1, "commentary": 2}
    for group in clusters:
        members = sorted(group, key=lambda x: (kind_order[x["event"]["source"]["kind"]],
                                                -_time(x["event"]["published_at"]).timestamp(), x["event"]["id"]))
        representative = members[0]["event"]
        member_ids = sorted(x["event"]["id"] for x in group)
        cluster_id = "nf-" + _hash(member_ids)[:16]
        families = sorted({_family(x["event"]["source"]) for x in group})
        for member in group:
            member["entry"]["cluster_id"] = cluster_id
            if member["entry"]["status"] == "candidate":
                member["entry"]["status"] = "representative" if member["event"]["id"] == representative["id"] else "repeated_wording"
        summaries.append({"cluster_id": cluster_id, "representative_id": representative["id"],
                          "title": representative["title"], "entities": list(representative["entities"]),
                          "published_at": representative["published_at"], "source": dict(representative["source"]),
                          "member_ids": member_ids, "report_count": len(group),
                          "declared_source_families": families, "independence_verified": False,
                          "truth_assessed": False, "impact_assessed": False,
                          "relevance": "unmapped" if wanted and not representative["entities"] else "matches_focus" if wanted else "not_filtered",
                          "event_keys": sorted({x["event"]["event_key"] for x in group if x["event"].get("event_key")})})
    # A shared event key with different wording may be a correction or conflict;
    # do not silently collapse it. All such groups remain individually visible.
    event_keys = {}
    for group in summaries:
        for key in group["event_keys"]:
            event_keys.setdefault(key, []).append(group["cluster_id"])
    conflicts = [{"event_key": key, "cluster_ids": values,
                  "reason": "different_wording_or_quantities_requires_review"}
                 for key, values in sorted(event_keys.items()) if len(values) > 1]
    conflict_ids = {cid for c in conflicts for cid in c["cluster_ids"]}
    summaries.sort(key=lambda x: (x["cluster_id"] not in conflict_ids,
                                   kind_order[x["source"]["kind"]],
                                   -_time(x["published_at"]).timestamp(), x["cluster_id"]))
    queue = summaries[:applied["max_items"]]
    candidate_count = len(candidates)
    return {
        "schema": "noisefloor.narrative-triage.v1", "as_of": as_of, "focus": sorted(wanted),
        "status": "triaged" if candidates else "insufficient_visibility",
        "policy": applied, "attention_queue": queue, "clusters": summaries,
        "events": sorted(classified, key=lambda e: e["id"]), "conflict_candidates": conflicts,
        "deferred_cluster_ids": [s["cluster_id"] for s in summaries[applied["max_items"]:]],
        "counts": {"supplied": len(events), "eligible_for_triage": candidate_count,
                   "distinct_wording_clusters": len(clusters), "repeated_wording": candidate_count-len(clusters),
                   "queued": len(queue), "deferred": max(0, len(clusters)-len(queue))},
        "limits": ["Wording similarity measures repetition, not truth, importance or independent corroboration.",
                   "Titles, entities, source kinds and source families are unverified caller data.",
                   "Headlines may omit decisive details; language and paraphrase coverage are limited.",
                   "Different numbers or explicit negation stay separate; this is not full contradiction detection.",
                   "An empty queue does not establish a quiet market; collection completeness is unknown.",
                   "Without availability timestamps this cannot establish an as-published historical record.",
                   "Review priority uses conflicts, source kind and recency, not expected returns."],
        "source_universe_complete": False, "execution_authority": False,
        "request_sha256": _hash(request), "policy_sha256": _hash(applied),
    }
