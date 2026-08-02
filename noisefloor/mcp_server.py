"""noisefloor MCP server — let an assistant check whether a number is real.

Assistants read metrics constantly now, and they assert significance the way
people do: by eyeballing a change and calling it. "Conversions are up 12% since
the deploy" is a claim about noise, and nothing in a language model checks it.

This exposes the noisefloor checks as tools so the assistant can stop guessing.
Stdio transport, JSON-RPC 2.0, standard library only — no dependencies to
install and nothing to configure.

Run:  python -m noisefloor.mcp_server
"""
from __future__ import annotations

import json
import sys

from . import change, coverage, experiment, forecast, multiple

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "noisefloor"
SERVER_VERSION = "0.2.0"

SERVER_INSTRUCTIONS = (
    "noisefloor answers one question: is this number real, or is it noise?\n\n"
    "USE THESE TOOLS BEFORE claiming that a metric changed, that an A/B test has "
    "a winner, or that a trend is meaningful. Eyeballing a percentage change is "
    "not evidence, and the usual statistics are invalid in the way people "
    "actually use them — a t-test assumes you looked once at a pre-committed "
    "sample size, and nobody does that.\n\n"
    "ab_test      — can a winner be called yet? Safe to run after every single\n"
    "               observation; peeking does not inflate the error rate.\n"
    "did_it_change— did a metric depart from its own history, up OR down?\n"
    "real_or_sampling — did the metric move, or did the sample under it move?\n"
    "               Run this before reporting any rate as a change.\n"
    "forecast_next— the next expected value with an honest range.\n"
    "score_forecasts — how well-calibrated past forecasts actually were.\n"
    "which_metrics_matter — given many metrics, which stand out once you\n"
    "               account for watching that many at once.\n\n"
    "Every result carries its method and its guarantee. Quote them: the point "
    "of these tools is that the claim can be backed, not just asserted."
)


def _series(arg, name: str) -> list[float]:
    if not isinstance(arg, list) or not arg:
        raise ValueError(f"{name} must be a non-empty list of numbers")
    out = []
    for v in arg:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        out.append(float(v))
    return out


def t_ab_test(a: dict) -> dict:
    return experiment.compare(
        int(a["a_successes"]), int(a["a_total"]),
        int(a["b_successes"]), int(a["b_total"]),
        alpha=float(a.get("alpha", experiment.DEFAULT_ALPHA)),
        labels=(str(a.get("a_label", "A")), str(a.get("b_label", "B"))))


def t_did_it_change(a: dict) -> dict:
    r = change.scan(_series(a.get("values"), "values"),
                    two_sided=bool(a.get("two_sided", True)),
                    half_life=a.get("half_life"))
    r.pop("states", None)
    r.pop("evidence_series", None)   # keep the payload small for an assistant
    return r


def t_real_or_sampling(a: dict) -> dict:
    return coverage.check(_series(a.get("values"), "values"),
                          _series(a.get("sample_sizes"), "sample_sizes"))


def t_forecast_next(a: dict) -> dict:
    return forecast.next_value(_series(a.get("values"), "values"),
                               nominal=float(a.get("nominal", forecast.NOMINAL)))


def t_score_forecasts(a: dict) -> dict:
    return forecast.score(_series(a.get("values"), "values"),
                          nominal=float(a.get("nominal", forecast.NOMINAL)))


def t_which_metrics_matter(a: dict) -> dict:
    ev = a.get("evidence")
    if not isinstance(ev, dict) or not ev:
        raise ValueError("evidence must be a non-empty object of metric -> number")
    return multiple.select({str(k): float(v) for k, v in ev.items()},
                           alpha=float(a.get("alpha", multiple.DEFAULT_ALPHA)))


_NUMS = {"type": "array", "items": {"type": "number"}}

TOOLS = {
    "ab_test": (
        "Can you call a winner on an A/B test yet? Uses anytime-valid confidence "
        "sequences, so it is SAFE TO RUN AFTER EVERY OBSERVATION — peeking does not "
        "inflate the false-positive rate the way a t-test or z-test does.",
        {"type": "object",
         "properties": {
             "a_successes": {"type": "integer", "description": "conversions in arm A"},
             "a_total": {"type": "integer", "description": "total observations in arm A"},
             "b_successes": {"type": "integer", "description": "conversions in arm B"},
             "b_total": {"type": "integer", "description": "total observations in arm B"},
             "alpha": {"type": "number", "description": "error budget, default 0.05"},
             "a_label": {"type": "string"}, "b_label": {"type": "string"}},
         "required": ["a_successes", "a_total", "b_successes", "b_total"],
         "additionalProperties": False},
        t_ab_test),
    "did_it_change": (
        "Did a metric actually change, or is the move noise? Detects both rises AND "
        "collapses against the metric's own history, with a stated false-alarm rate "
        "and no assumption about the distribution.",
        {"type": "object",
         "properties": {
             "values": dict(_NUMS, description="the metric's history, oldest first"),
             "two_sided": {"type": "boolean",
                           "description": "detect drops as well as rises (default true)"},
             "half_life": {"type": "number",
                           "description": "optional: readings after which old history "
                                          "counts half, for drifting metrics"}},
         "required": ["values"], "additionalProperties": False},
        t_did_it_change),
    "real_or_sampling": (
        "Did the metric move, or did the sample size underneath it move? Run this "
        "before reporting any RATE as a change — conversion rates, error rates and "
        "click-through all shift when the denominator shifts, for reasons that have "
        "nothing to do with the thing being measured.",
        {"type": "object",
         "properties": {
             "values": dict(_NUMS, description="the metric's history, oldest first"),
             "sample_sizes": dict(_NUMS,
                                  description="the denominator behind each reading, "
                                              "same order and length")},
         "required": ["values", "sample_sizes"], "additionalProperties": False},
        t_real_or_sampling),
    "forecast_next": (
        "What should the next reading be, and within what range? Range adapts to the "
        "metric's recent volatility and stays valid even when the metric shifts.",
        {"type": "object",
         "properties": {
             "values": dict(_NUMS, description="the metric's history, oldest first"),
             "nominal": {"type": "number", "description": "range coverage, default 0.8"}},
         "required": ["values"], "additionalProperties": False},
        t_forecast_next),
    "score_forecasts": (
        "How good would these forecasts actually have been? Grades every prediction "
        "the tool would have made over the history, using only what was known at the "
        "time, and reports calibration plus the worst misses.",
        {"type": "object",
         "properties": {
             "values": dict(_NUMS, description="the metric's history, oldest first"),
             "nominal": {"type": "number", "description": "range coverage, default 0.8"}},
         "required": ["values"], "additionalProperties": False},
        t_score_forecasts),
    "which_metrics_matter": (
        "You watch many metrics; which genuinely stand out? Controls the false "
        "discovery rate across all of them at once, which per-metric thresholds do "
        "not: forty metrics each alerting wrongly 5% of the time means two false "
        "alarms every round.",
        {"type": "object",
         "properties": {
             "evidence": {"type": "object",
                          "description": "metric name -> evidence value, e.g. the "
                                         "'evidence' field from did_it_change",
                          "additionalProperties": {"type": "number"}},
             "alpha": {"type": "number", "description": "false-discovery rate, default 0.1"}},
         "required": ["evidence"], "additionalProperties": False},
        t_which_metrics_matter),
}


# ------------------------------------------------------------------ protocol --
def _result(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


TOOL_TITLES = {
    "ab_test": "Peek-safe A/B verdict",
    "did_it_change": "Did this metric really change?",
    "real_or_sampling": "Real effect, or sampling noise?",
    "forecast_next": "Calibrated next-value forecast",
    "score_forecasts": "Score past forecasts honestly",
    "which_metrics_matter": "Rank metrics by real signal",
}

# Pure local computation on numbers the caller supplies: no state, no
# network, no side effects. Declared so cautious clients can auto-approve.
TOOL_ANNOTATIONS = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "destructiveHint": False,
    "openWorldHint": False,
}

# Prompts: playbooks MCP clients surface as slash commands. Each steers an
# agent to a defensible statistical verdict instead of an eyeballed one.
PROMPTS = {
    "ab_test_verdict": (
        "Is this A/B test actually done?",
        "A peek-safe verdict on an experiment from cumulative successes "
        "and trials, immune to the peeking that invalidates t-tests.",
        [],
        lambda a: (
            "Judge the A/B experiment I describe with the noisefloor "
            "tools: pass each arm's cumulative (successes, n) to ab_test "
            "and report its verdict exactly — 'decided' with the winner, "
            "or 'keep collecting' with the current evidence ratio. State "
            "why a plain t-test would be invalid here (continuous "
            "monitoring inflates false winners; the test martingale does "
            "not), and refuse to call a winner the tool has not called."
        ),
    ),
    "is_this_number_real": (
        "Is this number real, or is it noise?",
        "Route a suspicious metric movement through change detection and "
        "sampling-noise checks before anyone acts on it.",
        [],
        lambda a: (
            "For the metric movement I describe: 1) did_it_change on the "
            "series for a calibrated change verdict; 2) real_or_sampling "
            "on the before/after counts for whether sampling alone "
            "explains it; 3) which_metrics_matter if several metrics "
            "compete for attention. Report each verdict with its false "
            "alarm rate, and state plainly when the honest answer is "
            "'noise' or 'not enough data yet' — that is the product "
            "working, not failing."
        ),
    ),
}


def handle(msg: dict) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        return _result(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False},
                             "prompts": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION,
                           "title": "noisefloor — is this number real?",
                           "websiteUrl": "https://github.com/beepboop2025/noisefloor"},
            "instructions": SERVER_INSTRUCTIONS})
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": [
            {"name": n, "title": TOOL_TITLES.get(n, n), "description": d,
             "inputSchema": s,
             "annotations": {"title": TOOL_TITLES.get(n, n),
                             **TOOL_ANNOTATIONS}}
            for n, (d, s, _fn) in sorted(TOOLS.items())]})
    if method == "prompts/list":
        return _result(mid, {"prompts": [
            {"name": n, "title": t, "description": d, "arguments": args}
            for n, (t, d, args, _fn) in PROMPTS.items()]})
    if method == "prompts/get":
        params = msg.get("params") or {}
        name = params.get("name")
        entry = PROMPTS.get(name) if isinstance(name, str) else None
        if entry is None:
            return _error(mid, -32602, f"unknown prompt: {name}")
        _t, desc, _args_spec, fn = entry
        args = params.get("arguments")
        if not isinstance(args, dict):
            args = {}
        return _result(mid, {"description": desc, "messages": [
            {"role": "user", "content": {"type": "text", "text": fn(args)}}]})
    if method == "resources/list":
        return _result(mid, {"resources": []})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        if name not in TOOLS:
            return _error(mid, -32602, f"unknown tool: {name}")
        try:
            payload = TOOLS[name][2](params.get("arguments") or {})
        except (ValueError, KeyError, TypeError) as exc:
            # fail loud and legibly; never return a number we cannot stand behind
            return _result(mid, {
                "content": [{"type": "text",
                             "text": json.dumps({"error": str(exc)}, indent=2)}],
                "isError": True})
        return _result(mid, {"content": [
            {"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}]})
    return _error(mid, -32601, f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps(_error(None, -32700, "parse error")), flush=True)
            continue
        reply = handle(msg)
        if reply is not None:
            print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
